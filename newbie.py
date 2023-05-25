import asyncio
import numpy as np
import poke_env as poke_env
from gym.spaces import Space, Box
from gym.utils.env_checker import check_env
from tensorflow.keras.callbacks import TensorBoard
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from tabulate import tabulate
from poke_env.data import GenData, to_id_str

Data = GenData.from_gen(8)
Typechart = Data.type_chart
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from poke_env.environment.side_condition import SideCondition
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player import (
    background_evaluate_player,
    background_cross_evaluate,
    Gen8EnvSinglePlayer,
    RandomPlayer,
    MaxBasePowerPlayer,
    ObservationType,
    SimpleHeuristicsPlayer,
)

ENTRY_HAZARDS = {
        "spikes": SideCondition.SPIKES,
        "stealhrock": SideCondition.STEALTH_ROCK,
        "stickyweb": SideCondition.STICKY_WEB,
        "toxicspikes": SideCondition.TOXIC_SPIKES,
    }

ANTI_HAZARDS_MOVES = {"rapidspin", "defog"}

class SimpleRLPlayer(Gen8EnvSinglePlayer):
    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=100.0
        )
    # def calc_reward(self, last_battle, current_battle) -> float:
    #     return self.reward_computing_helper(
    #         current_battle, victory_value=30.0
    #     )
    def embed_battle(self, battle: AbstractBattle) -> ObservationType:
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart=Typechart,
                )
        moves_actual_power = [0] * 4
        for i, move in enumerate(battle.available_moves):
            moves_actual_power[i] = moves_base_power[i] * moves_dmg_multiplier[i] / 4
        
        #calculate the pokemon's resistance to the opponent's type choose max from two
        mon_rst_multiplier = max([
            battle.active_pokemon.damage_multiplier(
                battle.opponent_active_pokemon.type_1,
            ),
            battle.active_pokemon.damage_multiplier(
                battle.opponent_active_pokemon.type_2,
            ),
        ])

        # Show boost to env
        mon_boost = list(battle.active_pokemon._boosts.values())
        mon_boost = [boost for boost in mon_boost]
        # opponent_boost is no need
        # opponent_boost = list(battle.opponent_active_pokemon._boosts.values())
        # opponent_boost = [boost for boost in opponent_boost]

        # We count which pokemons I have fainted for switching purposes
        fainted_mon_team = [1]*6
        for i,mon in enumerate(battle.team.values()):
            if mon.fainted:
                fainted_mon_team[i] = 0
            
        
        # Show the current HP of active pokemon
        mon_hp = battle.active_pokemon.current_hp / 100
        opponent_hp = battle.opponent_active_pokemon.current_hp / 100

        mon_team_rst_multiplier = []
        for mon in battle.team.values():
            mon_team_rst_multiplier.append(max([
                mon.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                ),
                mon.damage_multiplier(
                    battle.opponent_active_pokemon.type_2,
                )
            ]))
        mon_active_index = -1
        for i, mon in enumerate(battle.team.values()):
            if mon == battle.active_pokemon:
                mon_active_index = i
                break
        mon_team_can_dynamax = int(battle.can_dynamax)
        mon_opponent_can_dynamax = int(battle.opponent_can_dynamax)
        mon_active_dynamaxed = int(battle.active_pokemon.is_dynamaxed)
        mon_opponent_dynamaxed = int(battle.opponent_active_pokemon.is_dynamaxed)
        n_opp_remaining_mons = 6 - len(
                [m for m in battle.opponent_team.values() if m.fainted is True]
            )
        
        mon_anti_hazard_move = -1
        for i,move in enumerate(battle.available_moves):
            if move.id in ANTI_HAZARDS_MOVES:
                mon_anti_hazard_move = i
                break
        mon_entry_hazard_move = -1
        for i,move in enumerate(battle.available_moves):
            if move.id in ENTRY_HAZARDS:
                mon_entry_hazard_move = i
                break
        mon_team_side_conditions = 0
        for condition in ENTRY_HAZARDS.values():
            if condition in battle.side_conditions:
                mon_team_side_conditions = 1
                break
        mon_opponent_side_conditions = 0
        for condition in ENTRY_HAZARDS.values():
            if condition in battle.opponent_side_conditions:
                mon_opponent_side_conditions = 1
                break
        mon_spe = battle.active_pokemon.base_stats["spe"] / 100
        mon_opponent_spe = battle.opponent_active_pokemon.base_stats["spe"] / 100
        
        
        # Final vector with 10 components
        final_vector = np.concatenate(
            [
                [mon_active_index],# 1
                [mon_rst_multiplier],# 1
                mon_team_rst_multiplier,# 6
                [mon_hp,opponent_hp],# 2 * 1
                [mon_team_can_dynamax,
                mon_opponent_can_dynamax,
                mon_active_dynamaxed,
                mon_opponent_dynamaxed],#4 * 1
                moves_actual_power,# 4
                fainted_mon_team, # 6
                [n_opp_remaining_mons],# 1
                [mon_anti_hazard_move,
                 mon_entry_hazard_move],# 2
                 [mon_team_side_conditions,
                mon_opponent_side_conditions],# 2
                [mon_spe,mon_opponent_spe]# 2 * 1
            ]
        )
        return np.float32(final_vector)

    def describe_embedding(self) -> Space:
        # low = [0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # high = [1, 1, 1, 1, 3, 3, 3, 3, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        low = [0] * 1 + [0] * 1 + [0] * 6 + [0] * 2 + [0] * 4 + [-1] * 4  + [0] * 6 + [0] * 1 + [-1] * 2 + [0] * 2 + [0] * 2
        high = [5] * 1 + [4] * 1 + [4] * 6 + [8] * 2 + [1] * 4 + [3] * 4  + [1] * 6 + [6] * 1 + [3] * 2 + [1] * 2 + [4] * 2
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )


async def train_with_opponent(opponent_type):
    if opponent_type == "random":
        opponent = RandomPlayer(battle_format="gen8randombattle")
        train_env = SimpleRLPlayer(
            battle_format="gen8randombattle", opponent=opponent, start_challenging=True
        )
        
    elif opponent_type == "max":
        opponent = MaxBasePowerPlayer(battle_format="gen8randombattle")
        train_env = SimpleRLPlayer(
            battle_format="gen8randombattle", opponent=opponent, start_challenging=True
        )   
    elif opponent_type == "heuristic":
        opponent = SimpleHeuristicsPlayer(battle_format="gen8randombattle")
        train_env = SimpleRLPlayer(
            battle_format="gen8randombattle", opponent=opponent, start_challenging=True
        )
    opponent = RandomPlayer(battle_format="gen8randombattle")
    eval_env = SimpleRLPlayer(
        battle_format="gen8randombattle", opponent=opponent, start_challenging=True
    )
    # Compute dimensions
    n_action = train_env.action_space.n
    input_shape = (1,) + train_env.observation_space.shape
    print(input_shape)
    # Create model
    model = Sequential()
    model.add(Dense(256, activation="elu", input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(64, activation="elu"))
    model.add(Dense(n_action, activation="linear"))

    # Defining the DQN
    memory = SequentialMemory(limit=3600, window_length=1)

    # policy = LinearAnnealedPolicy(
    #     EpsGreedyQPolicy(),
    #     attr="eps",
    #     value_max=0.8,
    #     value_min=0.05,
    #     value_test=0.0,
    #     nb_steps=50000,
    # )
    policy = EpsGreedyQPolicy(eps=0.1)

    dqn = DQNAgent(
        model=model,
        nb_actions=n_action,
        policy=policy,
        memory=memory,
        nb_steps_warmup=1000,
        gamma=0.95,
        target_model_update=1,
        delta_clip=0.01,
        enable_double_dqn=True,
        enable_dueling_network=True,
    )
    dqn.compile(Adam(learning_rate=0.00025), metrics=["mae"])
    dqn.load_weights("./newbie/130000.h5")
    tb = TensorBoard(log_dir="./logs/newbie/140000")
    dqn.fit(train_env, nb_steps=10000, callbacks=[tb])
    train_env.close()
    dqn.save_weights("./newbie/140000.h5", overwrite=True)

    print("Results against random player:")
    dqn.test(eval_env, nb_episodes=10, verbose=False, visualize=False)
    print(
        f"DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes"
    )
    second_opponent = MaxBasePowerPlayer(battle_format="gen8randombattle")
    eval_env.reset_env(restart=True, opponent=second_opponent)
    print("Results against max base power player:")
    dqn.test(eval_env, nb_episodes=10, verbose=False, visualize=False)
    print(
        f"DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes"
    )
    eval_env.reset_env(restart=False)
    eval_env.close()

async def evaluate():
    opponent = RandomPlayer(battle_format="gen8randombattle")
    eval_env = SimpleRLPlayer(
        battle_format="gen8randombattle", opponent=opponent, start_challenging=True
    )
    # Compute dimensions
    n_action = eval_env.action_space.n
    input_shape = (1,) + eval_env.observation_space.shape

    # Create model
    model = Sequential()
    model.add(Dense(256, activation="elu", input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(64, activation="elu"))
    model.add(Dense(n_action, activation="linear"))

    # Defining the DQN
    memory = SequentialMemory(limit=3600, window_length=1)

    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0.0,
        nb_steps=10000,
    )

    dqn = DQNAgent(
        model=model,
        nb_actions=n_action,
        policy=policy,
        memory=memory,
        nb_steps_warmup=1000,
        gamma=0.95,
        target_model_update=1,
        delta_clip=0.01,
        enable_double_dqn=True,
        enable_dueling_network=True,
    )
    dqn.compile(Adam(learning_rate=0.00025), metrics=["mae"])
    # dqn.load_weights("d3qn.h5")
    # Training the model
    dqn.load_weights("./newbie/140000.h5")
    print("Results against random player:")
    dqn.test(eval_env, nb_episodes=100, verbose=False, visualize=False)
    print(
        f"DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes"
    )
    second_opponent = MaxBasePowerPlayer(battle_format="gen8randombattle")
    eval_env.reset_env(restart=True, opponent=second_opponent)
    print("Results against max base power player:")
    dqn.test(eval_env, nb_episodes=100, verbose=False, visualize=False)
    print(
        f"DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes"
    )
    eval_env.reset_env(restart=False)
    eval_env.close()

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(train_with_opponent("max"))
    # asyncio.get_event_loop().run_until_complete(evaluate())
