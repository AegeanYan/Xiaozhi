import asyncio
import numpy as np

from gym.spaces import Space, Box
from stable_baselines3 import DQN
from tabulate import tabulate
from stable_baselines3.common.evaluation import evaluate_policy

from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.data import GenData 
from poke_env.player import (
    background_evaluate_player,
    background_cross_evaluate,
    Gen8EnvSinglePlayer,
    RandomPlayer,
    MaxBasePowerPlayer,
    ObservationType,
    SimpleHeuristicsPlayer,
)
Typechart = GenData.from_gen(8).type_chart

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

        # We count how many pokemons have fainted in each team
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        final_vector = np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [fainted_mon_team, fainted_mon_opponent],
            ]
        )
        return np.float32(final_vector)

    def describe_embedding(self) -> Space:
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )


async def main():
    # First test the environment to ensure the class is consistent
    # with the OpenAI API

    # opponent = RandomPlayer(battle_format="gen8randombattle")
    # test_env = SimpleRLPlayer(battle_format="gen8randombattle",opponent=opponent, start_challenging=True)
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

    # Defining the DQN
    # model = DQN(
    #     "MlpPolicy",
    #     env=train_env,
    #     learning_rate=5e-5,
    #     buffer_size=50000, 
    #     verbose=1,
    #     device="cpu",
    #     tensorboard_log="./dqn_log/",
    # )

    # model.learn(total_timesteps=5)
    # # Evaluating the model
    # train_env.close()

    # model.save("dqn_5")

    # del model

    model = DQN.load("dqn_50000")

    # print("Results against random player:")
    # evaluate_policy(model, eval_env, n_eval_episodes=10, render=False)
    # print(
    #     f"DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes"
    # )
    # second_opponent = MaxBasePowerPlayer(battle_format="gen8randombattle")
    # eval_env.reset_env(restart=True, opponent=second_opponent)
    # print("Results against max base power player:")
    # evaluate_policy(model, eval_env, n_eval_episodes=10, render=False)
    # print(
    #     f"DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes"
    # )
    # eval_env.reset_env(restart=False)

    # Evaluate the player with included util method
    # n_challenges = 25
    # placement_battles = 4
    # eval_task = background_evaluate_player(
    #     eval_env.agent, n_challenges, placement_battles
    # )
    # evaluate_policy(model, eval_env, n_eval_episodes=n_challenges, render=False)
    # print("Evaluation with included method:", eval_task.result())
    eval_env.reset_env(restart=False)

    # Cross evaluate the player with included util method
    n_challenges = 5
    players = [
        eval_env.agent,
        RandomPlayer(battle_format="gen8randombattle"),
        MaxBasePowerPlayer(battle_format="gen8randombattle"),
        SimpleHeuristicsPlayer(battle_format="gen8randombattle"),
    ]
    cross_eval_task = background_cross_evaluate(players, n_challenges)
    # evaluate_policy(model, eval_env, n_eval_episodes=n_challenges*(len(players)-1), render=False)
    cross_evaluation = cross_eval_task.result()
    table = [["-"] + [p.username for p in players]]
    for p_1, results in cross_evaluation.items():
        table.append([p_1] + [cross_evaluation[p_1][p_2] for p_2 in results])
    print("Cross evaluation of DQN with baselines:")
    print(tabulate(table))
    eval_env.close()


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())