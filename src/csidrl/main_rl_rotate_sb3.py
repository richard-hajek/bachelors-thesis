import copy
import json
import os
import time
from datetime import datetime, timedelta

import torch
import numpy as np
import IPython
import gym
from gym.wrappers import FlattenObservation
from matplotlib import pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter

import csidrl.environments.hydraulic_rotations
import csidrl.environments.wrappers
from csidrl import meowtools, optimisation, visualisation
from csidrl.agents.agents_neural_sb3 import DQNAgentWrapper
from csidrl.deep.sb3_compat_satnet import SATNetFeaturesExtractor
from csidrl.environments.hydraulic_rotations import HydraulicRotationsEnv
from csidrl.environments.wrappers import CompatibilityWrapper
from csidrl.main_rl_rotate import get_latest_agent
from csidrl.meowtools import RL_eval
from csidrl.monitor import SaveOnBestTrainingRewardCallback

from csidrl.discord_control.discordbot import send_message

log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)


def stop_RL_training(mean_reward, acceptable_deviation=1):

    if len(mean_reward) >= 6 and np.mean(mean_reward[-6:]) <= acceptable_deviation:
        return True

    if len(mean_reward) >= 12 and np.abs(np.mean(mean_reward[-6:]) - np.mean(mean_reward[-12:])) < acceptable_deviation:
        return True

    if len(mean_reward) > 16 and np.mean(mean_reward[-8:]) > np.mean(mean_reward[-16:-8]):
        return True

    return False


def create_env(compat=True, flatten=True, compact=True):
    env = HydraulicRotationsEnv(size=3, checkpoints=2, extended_reward=False)
    env = gym.wrappers.TimeLimit(env=env, max_episode_steps=200)

    if compact:
        env = csidrl.environments.hydraulic_rotations.RotationsBetterment(env)

    if flatten:
        env = FlattenObservation(env)

    if compat:
        env = CompatibilityWrapper(env)

    env = Monitor(env, log_dir, allow_early_resets=True)

    #check_env(env)
    return env


def create_agent_legacy(env, verbose=1, policy_kwargs=None, exploration_fraction=0.9):

    if policy_kwargs is None:
        policy_kwargs = {
            "features_extractor_class": SATNetFeaturesExtractor,
            "net_arch": [25, ],
        }

    model = DQN(
        "MlpPolicy",
        env,
        verbose=verbose,
        policy_kwargs=policy_kwargs,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        learning_starts=10_000,
        buffer_size=10_000,
    )
    return model


def build_agent(env, total_timesteps, iterations, rl_iterations, **kwargs):
    return DQN(env=env, **kwargs, verbose=1)

#@optimisation.cache.memoize(name="estimate_rotate_2", ignore={"env"})
#@optimisation.cache.memoize(name="estimate_rotate_2_env_10", ignore={"env"})
def estimate(env, total_timesteps, iterations, rl_iterations, **kwargs):

    callback = SaveOnBestTrainingRewardCallback(
        check_freq=100, log_dir=log_dir, verbose=0
    )
    start = time.time()
    eval_results = []
    eval_stds = []

    for _ in range(rl_iterations):
        agent = DQN(env=env, **kwargs, verbose=1)

        for _ in range(iterations):
            agent.learn(total_timesteps=total_timesteps, log_interval=4, callback=callback)

        score, std = evaluate_policy(agent, env)
        eval_results.append(score)
        eval_stds.append(std)

    end = time.time()

    mean_score = np.mean(eval_results)
    mean_std = np.mean(eval_stds)

    return (mean_score, mean_std), end - start


class Logger:

    instance = None

    def __init__(self, path="./log.jsonl"):
        self.path = path

    @staticmethod
    def log(obj):

        if Logger.instance is None:
            Logger.instance = Logger()

        Logger.instance._log(obj)

    def _log(self, obj):
        with open(self.path, "a") as f:
            json.dump(obj, f, default=str)
            f.write("\n")


class Response:

    def __init__(self, goal):
        self.goal = goal
        self.prev_best = float("-inf")

    def respond(self, e):

        r = ""

        if self.prev_best == float("-inf"):
            r += "- Initial measurement ;\n"

        if e > self.prev_best:
            r += "- New best measurement ;\n"

        if e >= self.goal and e > self.prev_best:
            r += "- New successful measurement! 🥳🥳\n"

        if not r:
            r += "- Nothing noteworthy"

        self.prev_best = max(e, self.prev_best)

        return r


response = Response(goal=-10)


def estimation_callback(kwargs, estimated, i, i_max):
    (score, std), t = estimated

    kwargs = copy.copy(kwargs)
    kwargs.pop("env")

    message = f"""
**Step {i}/{i_max}**
For kwargs: ```json
{json.dumps(kwargs, indent=4, skipkeys=True, default=lambda x: "[Object object]")}
```
Reached result:
- score: `{score:.2f}`
- std: `{std:.2f}`
- time: `{str(timedelta(seconds=t))}`

Automated scoring system:
{response.respond(score)}
    """

    send_message(message)
    Logger.log((kwargs, estimated, i, i_max))


history = []


def dummy_gridsearch(*args, **kwargs):
    history.append((kwargs, (("a", "b"), "c"), 0, 0))
    return (-100, 0), 0


def gridsearch(env, dummy=False):
    options = dict(
        # Reinforcement Learning opts
        env=[env],
        iterations=[10],
        rl_iterations=[1],

        # DQN Alg opts
        total_timesteps=[20_000],
        buffer_size=[100_000],
        learning_starts=[5_000],
        target_update_interval=[1_000],
        gamma=[0.99],
        tau=[1],
        #train_freq=[4],
        train_freq=[32],
        exploration_fraction=[0.5, 0.8],
        exploration_initial_eps=[1.0],
        exploration_final_eps=[0.05],

        # Neural network params
        policy=["MlpPolicy"],
        policy_kwargs=[{"features_extractor_class": SATNetFeaturesExtractor,"net_arch": [25, ],}, {"features_extractor_class": SATNetFeaturesExtractor,"net_arch": [9, ],}],
        learning_rate=[1e-3, 1e-4, 1e-5],
        batch_size=[8],
    )

    send_message("**Beginning scanning 🔎**")

    try:

        if not dummy:
            # Write all things to history
            #optimisation.gridsearch(dummy_gridsearch, options)
            results = optimisation.gridsearch(estimate, options, estimation_callback=estimation_callback)
        else:
            results = optimisation.gridsearch(dummy_gridsearch, options)

        print(results)

        if len(results) >= 35:
            visualisation.show(results, "c1", "c2", lambda x: x["estimate"] > 5)

    except Exception as e:
        send_message(f"Automated scan crashed unexpectedly: \"*{e}*\"")
        raise

    send_message("Scanning ended")


def show():
    # Helper from the library
    results_plotter.plot_results(
        [log_dir], 1e5, results_plotter.X_TIMESTEPS, "Rotations"
    )

    plt.show()


def RL_SB3_eval(agent, env):
    env = create_env(compat=False)
    RL_eval(agent, env)


def main():
    env = create_env(flatten=False, compact=False)
    eval_env = create_env(flatten=False, compact=False)
    agent = create_agent_legacy(env)
    #agent=None

    #callback = SaveOnBestTrainingRewardCallback(
    #    check_freq=100, log_dir=log_dir, verbose=0
    #)

    env.reset()

    # Train
    actions = {
        "ipy": lambda agent=agent, env=env, meowtools=meowtools, torch=torch, np=np: IPython.embed(),
        "eval": lambda: RL_SB3_eval(DQNAgentWrapper(agent), env),
        "train": lambda: agent.learn(
            total_timesteps=50_000, log_interval=4, eval_freq=1000, eval_env=eval_env,
        ),
        "grid": lambda: gridsearch(env),
        "dgrid": lambda: gridsearch(env, dummy=True),
        "show": lambda: show(),
        "save": lambda: agent.save(
            f"agents/rl_rotate_sb3.{datetime.today()}.agent"
        ),
        "load": lambda: agent.set_parameters(get_latest_agent("rl_rotate_sb3")),
    }

    meowtools.console(actions)


if __name__ == "__main__":

    main()
