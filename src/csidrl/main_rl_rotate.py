import time
from datetime import datetime

import IPython
import gym
import numpy as np
from torch import optim, nn

from csidrl import meowtools, optimisation, visualisation
from csidrl.agents import agents_neural
from csidrl.agents.agents import QLearningAgent
from csidrl.agents.agents_neural import MeowQNetworkAgent
from csidrl.environments.hydraulic_rotations import HydraulicRotationsEnv
from csidrl.deep.meownets import MeowModule
from csidrl.deep import meownets

import glob
import os



def create_env():
    env = HydraulicRotationsEnv(size=4, checkpoints=2)
    env = gym.wrappers.TimeLimit(env=env, max_episode_steps=100)
    return env


def create_agent(
    env: gym.Env,
    dl_batch_size=512,
    dl_learning_rate=0.5,
    dl_loss=nn.MSELoss,
    dl_optimizer=optim.Adam,
    rl_discount_factor=0.99,
    rl_dl_max_dataset=10_000,
    rl_dl_rate=1000,
    rl_dl_target_update_rate=2000,
    rl_exploration_factor=0.5,
    rl_learning_rate=0.5,
    rl_replay_buffer_size=100_000,
):
    q_network = MeowModule(
        nn_module=meownets.DeepLeakyRelu,
        input_shape=env.observation_space.shape,
        output_shape=(env.action_space.n,),
        optimizer=dl_optimizer,
        loss=dl_loss,
        batch_size=dl_batch_size,
        lr=dl_learning_rate,
    )

    agent = agents_neural.MeowQNetworkAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        q_network=q_network,
        discount_factor=rl_discount_factor,
        exploration_factor=rl_exploration_factor,
        buffer_size=rl_replay_buffer_size,
        dl_train_rate=rl_dl_rate,
        dl_target_update_rate=rl_dl_target_update_rate,
        max_dataset_size=rl_dl_max_dataset,
        learning_rate=rl_learning_rate,
    )

    return agent


def get_latest_agent(name="rl_rotate"):
    list_of_files = glob.glob(f'./agents/{name}*')
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


@optimisation.cache.memoize(name="estimate_gridsearch", ignore={"env"})
def estimate(env, **kwargs):
    start = time.time()
    agent = create_agent(env, **kwargs)
    meowtools.RL_train(agent, env, 100, allow_update=True, verbose=True)
    eval_result = meowtools.RL_eval(agent, env, episodes=100)
    eval_result = np.mean(eval_result)
    end = time.time()
    return eval_result, end - start


def gridsearch(env):
    options = dict(
        env=[env],
        dl_batch_size=[512],
        dl_learning_rate=[0.5, 0.9, 0.1],
        dl_loss=[nn.MSELoss],
        dl_optimizer=[optim.Adam],
        rl_discount_factor=[0.99, 0.5, 0.8],
        rl_dl_max_dataset=[10_000],
        rl_dl_rate=[500],
        rl_dl_target_update_rate=[1000],
        rl_exploration_factor=[0.5, 0.1],
        rl_learning_rate=[0.5, 0.1],
        rl_replay_buffer_size=[100_000]
    )

    results = optimisation.gridsearch(estimate, options)
    visualisation.show(results, "c1", "c2", lambda x: x["estimate"] > -10)


def ttc_loop(agent, env):
    meowtools.RL_train(agent, env, 2000, allow_update=True, verbose=True)


def main():

    # Creation
    env = create_env()
    #agent = create_agent(env)
    agent = create_agent(env)

    actions = {
        "ipy": lambda agent=agent, env=env, meowtools=meowtools: IPython.embed(),
        "train": lambda: meowtools.RL_train(agent, env, 1, allow_update=False),
        "copy": lambda: agent.update_target(),
        "think": lambda: agent.DL_train(epochs=100, verbose=True),
        "eval": lambda: meowtools.RL_eval(agent, env),
        "save": lambda: agent.save(f"agents/rl_rotate.{datetime.today()}.agent"),
        "load": lambda: agent.load(get_latest_agent("rl_rotate")),
        "loop": lambda: ttc_loop(agent, env),
        "grid": lambda: gridsearch(env),
    }

    meowtools.console(actions)


if __name__ == "__main__":
    main()
