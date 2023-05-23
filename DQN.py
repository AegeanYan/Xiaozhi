import asyncio
import numpy as np
import poke_env as poke_env
from gym.spaces import Space, Box
from gym.utils.env_checker import check_env
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
from tensorflow.keras.optimizers.legacy import Adam

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


class SimpleRLPlayer(Gen8EnvSinglePlayer):
    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

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
        #calculate the pokemon's resistance to the opponent's type
        mon_rst_multiplier = [
            battle.active_pokemon.damage_multiplier(
                battle.opponent_active_pokemon.type_1,
            ),
            battle.active_pokemon.damage_multiplier(
                battle.opponent_active_pokemon.type_2,
            ),
        ]
        # Show pokemon id to env
        mon_team_id = [Data.pokedex[mon._species]['num']/1000 for mon in battle.team.values()]
        mon_opponent_id = [Data.pokedex[mon._species]['num']/1000 for mon in battle.opponent_team.values()]
        while len(mon_opponent_id) < 6:
            mon_opponent_id.append(0)

        # Show boost to env
        mon_boost = list(battle.active_pokemon._boosts.values())
        mon_boost = [boost/6 for boost in mon_boost]
        opponent_boost = list(battle.opponent_active_pokemon._boosts.values())
        opponent_boost = [boost/6 for boost in opponent_boost]

        # We count how many pokemons have fainted in each team
        fainted_mon_team = [1]*6
        for i,mon in enumerate(battle.team.values()):
            if mon.fainted:
                fainted_mon_team[i] = 0
        fainted_mon_opponent = [1]*6
        for i,mon in enumerate(battle.opponent_team.values()):
            if mon.fainted:
                fainted_mon_opponent[i] = 0
            
        
        # Show the current HP of active pokemon
        mon_hp = battle.active_pokemon.current_hp / battle.active_pokemon.max_hp
        opponent_hp = battle.opponent_active_pokemon.current_hp / battle.opponent_active_pokemon.max_hp

        # Show current mon
        mon_current = Data.pokedex[battle.active_pokemon._species]['num']/1000
        opponent_current = Data.pokedex[battle.opponent_active_pokemon._species]['num']/1000
        # Final vector with 10 components
        final_vector = np.concatenate(
            [
                [mon_current, opponent_current],#2 * 1
                [mon_hp,opponent_hp],# 2 * 1
                moves_base_power,# 4
                moves_dmg_multiplier,# 4
                fainted_mon_team, # 6
                fainted_mon_opponent,# 6
                mon_rst_multiplier,# 2
                mon_boost,# 7
                opponent_boost,# 7
                mon_team_id,# 6
                mon_opponent_id,# 6
            ]
        )
        return np.float32(final_vector)

    def describe_embedding(self) -> Space:
        low = [0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        high = [1, 1, 1, 1, 3, 3, 3, 3, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )


async def main():
    # First test the environment to ensure the class is consistent
    # with the OpenAI API
    # test_env = SimpleRLPlayer(battle_format="gen8randombattle", start_challenging=True)
    # check_env(test_env)
    # test_env.close()

    # Create one environment for training and one for evaluation
    opponent = RandomPlayer(battle_format="gen8randombattle")
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

    # Create model
    model = Sequential()
    model.add(Dense(128, activation="elu", input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(64, activation="elu"))
    model.add(Dense(n_action, activation="linear"))

    # Defining the DQN
    memory = SequentialMemory(limit=2000, window_length=1)

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
        gamma=0.5,
        target_model_update=1,
        delta_clip=0.01,
        enable_double_dqn=True,
        enable_dueling_network=True,
    )
    dqn.compile(Adam(learning_rate=0.00025), metrics=["mae"])

    # Training the model
    dqn.fit(train_env, nb_steps=10000)
    train_env.close()
    dqn.save_weights("d3qn.h5", overwrite=True)

    # dqn.load_weights("dqn.h5")

    # Evaluating the model
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

    # Evaluate the player with included util method
    n_challenges = 250
    placement_battles = 40
    eval_task = background_evaluate_player(
        eval_env.agent, n_challenges, placement_battles
    )
    dqn.test(eval_env, nb_episodes=n_challenges, verbose=False, visualize=False)
    print("Evaluation with included method:", eval_task.result())
    eval_env.reset_env(restart=False)

    # Cross evaluate the player with included util method
    n_challenges = 50
    players = [
        eval_env.agent,
        RandomPlayer(battle_format="gen8randombattle"),
        MaxBasePowerPlayer(battle_format="gen8randombattle"),
        SimpleHeuristicsPlayer(battle_format="gen8randombattle"),
    ]
    cross_eval_task = background_cross_evaluate(players, n_challenges)
    dqn.test(
        eval_env,
        nb_episodes=n_challenges * (len(players) - 1),
        verbose=False,
        visualize=False,
    )
    cross_evaluation = cross_eval_task.result()
    table = [["-"] + [p.username for p in players]]
    for p_1, results in cross_evaluation.items():
        table.append([p_1] + [cross_evaluation[p_1][p_2] for p_2 in results])
    print("Cross evaluation of DQN with baselines:")
    print(tabulate(table))
    eval_env.close()


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
