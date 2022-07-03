import IPython
import gym
import numpy as np

from csidrl import meowtools
from csidrl.agents import agents
from csidrl.environments import wrappers


def render_episode(agent):
    env = gym.make("CartPole-v1", render_mode="human")
    env = wrappers.DiscretizedCartPole(env, n_bins=16)
    ret = meowtools.train_episode(env, agent, evalution=True)
    env.close()

    print(f"Rendered return: {ret}")


def make_env():
    env = gym.make("CartPole-v1")
    env.reset(seed=42)
    env = wrappers.DiscretizedCartPole(env, n_bins=16)
    env = wrappers.RecordWraper(env)
    return env


def main():
    env = make_env()

    agent = agents.QLearningAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        exploration_factor=0.5,
        weight_initializer=lambda x: np.zeros(x),
        learning_rate=0.5,
    )

    actions = {
        "ipy": lambda agent=agent, env=env, meowtools=meowtools: IPython.embed(),
        "train": lambda: meowtools.train_episode(env, agent),
        "render": lambda: render_episode(agent),
    }

    meowtools.console(actions)


if __name__ == "__main__":
    main()
