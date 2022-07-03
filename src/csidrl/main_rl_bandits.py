from datetime import datetime

import IPython
import gym
import numpy as np
from torch import optim, nn

from csidrl import meowtools
from csidrl.agents import agents_neural
from csidrl.deep import meownets
from csidrl.deep.meownets import MeowModule
from csidrl.environments.bandit_env import BanditTenArmedUniformDistributedReward
from csidrl.environments.wrappers import BoxidizerWrapper
from csidrl.main_rl_rotate import get_latest_agent


def create_env():
    bandits = BanditTenArmedUniformDistributedReward()
    bandits = BoxidizerWrapper(bandits)
    return bandits


def create_agent(env: gym.Env):

    q_network = MeowModule(
        nn_module=meownets.DeepLeakyReluSigmoidOut,
        input_shape=env.observation_space.shape,
        output_shape=(env.action_space.n, ),
        optimizer=optim.Adam,
        loss=nn.MSELoss,
        batch_size=32,
        lr=0.5,
    )

    agent = agents_neural.MeowQNetworkAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        q_network=q_network,
        discount_factor=0.99,
        exploration_factor=0.5,
        buffer_size=100_000,
    )

    return agent


def ttc_loop(agent, env):

    meowtools.RL_train(agent, env, 10000, allow_update=True, verbose=True)


def main():

    env = create_env()
    agent = create_agent(env)

    # Train
    actions = {
        "ipy": lambda agent=agent, env=env, meowtools=meowtools: IPython.embed(),
        "train": lambda: meowtools.RL_train(agent, env, 1, allow_update=False),
        "think": lambda: agent.DL_train(epochs=100, verbose=True),
        "eval": lambda: meowtools.RL_eval(agent, env),
        "copy": lambda: agent.update_target(),
        "save": lambda: agent.save(f"agents/rl_bandits.{datetime.today()}.agent"),
        "load": lambda: agent.load(get_latest_agent(name="rl_bandits")),
        "loop": lambda: ttc_loop(agent, env),
    }

    meowtools.console(actions)


if __name__ == "__main__":
    main()