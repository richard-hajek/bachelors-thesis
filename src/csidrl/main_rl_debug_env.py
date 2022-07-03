from datetime import datetime

import IPython
import numpy as np
from torch import optim, nn

from csidrl import meowtools
from csidrl.agents import agents, agents_neural
from csidrl.deep.meownets import MeowModule
from csidrl.environments import wrappers, debug_envs
from csidrl.agents.agents_neural import MeowQNetworkAgent
from csidrl.deep import meownets

def make(**kwargs):
    env = debug_envs.GoRightEnv(states=10, limit=300, make_box=True)
    env = wrappers.TrackingWrapper(env)
    return env


def make_agent(env):
    agent = agents.QLearningAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        exploration_factor=0.5,
        weight_initializer=lambda x: np.random.random(size=x) + 10,
    )

    return agent


def make_dl_agent(env):
    q_network = MeowModule(
        nn_module=meownets.LinearRelu,
        input_shape=env.observation_space.shape,
        output_shape=(env.action_space.n,),
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

def main():
    env = make()
    agent = make_dl_agent(env)

    actions = {
        "ipy": lambda agent=agent, env=env, meowtools=meowtools: IPython.embed(),
        "train": lambda: meowtools.RL_train(agent, env, 1, allow_update=False), #meowtools.train_episode(env, agent),
        "dltrain": lambda: agent.update(0, epochs=100, verbose=True),
        "eval": lambda: meowtools.train_episode(env, agent, evalution=True),
        "save": lambda: agent.save(f"agents/rl_rotate.{datetime.today()}.agent"),
        "inspect": lambda: meowtools.train_episode(
            env, agent, evalution=True, debug=True
        ),
        "report": lambda: meowtools.render_progress(),
    }

    meowtools.console(actions)


if __name__ == "__main__":
    main()
