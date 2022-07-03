import time

import IPython
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

from csidrl import meowtools, optimisation, visualisation
from csidrl.environments.bandit_env import BanditTenArmedUniformDistributedReward
from csidrl.environments.wrappers import BoxidizerWrapper, DelayedReward, CompatibilityWrapper


def create_env():
    bandits = BanditTenArmedUniformDistributedReward()
    # bandits = BoxidizerWrapper(bandits)
    bandits = DelayedReward(bandits, 10)
    bandits = CompatibilityWrapper(bandits)
    check_env(bandits)
    return bandits


def create_agent(env):
    model = DQN("MlpPolicy", env, verbose=1, buffer_size=1_000, learning_starts=10_000, learning_rate=0.01)
    return model


@optimisation.cache.memoize(ignore={"env"})
def estimate(env, total_timesteps, **kwargs):
    start = time.time()
    agent = DQN(env=env, **kwargs, verbose=1)
    agent.learn(total_timesteps=total_timesteps, log_interval=10)
    eval_result = evaluate_policy(agent, env)
    end = time.time()
    return eval_result, end - start


def gridsearch(env):
    options = dict(

        # Reinforcement Learning opts
        env=[env],
        policy=["MlpPolicy"],

        # DQN Alg opts
        total_timesteps=[20_000],
        buffer_size=[10_000],
        learning_starts=[2_000],
        target_update_interval=[1000],
        gamma=[0.9, 0.99],
        tau=[1],
        train_freq=[4, 16, 32],
        exploration_fraction=[0.1],
        exploration_initial_eps=[1.0],
        exploration_final_eps=[0.05],

        # Neural network params
        learning_rate=[1e-4, 1e-5, 1e-6],
        batch_size=[32, 64, 128],
    )

    results = optimisation.gridsearch(estimate, options)
    visualisation.show(results, "c1", "c2", lambda x: x["estimate"] > 5,
                       weights={"estimate":10})


def main():
    env = create_env()
    agent = create_agent(env)

    # Train
    actions = {
        "ipy": lambda agent=agent, env=env, meowtools=meowtools: IPython.embed(),
        "train": lambda: agent.learn(total_timesteps=50_000, log_interval=4),
        "grid": lambda: gridsearch(env),
    }

    meowtools.console(actions)


if __name__ == "__main__":
    main()
