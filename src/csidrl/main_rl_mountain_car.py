import IPython
import gym
import numpy as np

import csidrl.legacy.agents_neural_legacy
from csidrl import meowtools
from csidrl.agents import agents
from csidrl.environments import wrappers


def create_car_env(
    render_mode=None,
    bins=None,
    extended=False,
    debug=False,
    continuous=False,
    discretizate=True,
    hist=False,
):
    env = gym.make("MountainCar-v0", render_mode=render_mode)

    if extended:
        env = gym.make("MountainCar1000-v0")

    if hist:
        env = wrappers.HistogramWrapper(env)

    if continuous:
        env = wrappers.ContinousRewardWrapper(env)

    if discretizate:
        env = wrappers.DiscretizedMountainCar(env, n_bins=bins)

    if debug:
        env = wrappers.ActionPrintingCarWrapper(env)

    return env


def render_episode(agent, bins, seed=None):
    env = create_car_env("rgb_array", bins)
    env = wrappers.GifRenderer(env, "./rendered.gif")
    ret = meowtools.train_episode(env, agent, render=True, evalution=True, seed=seed)
    # env.close()
    env.reset()
    print(f"Score: {ret}")


def main():
    # Args
    bins = 16

    # Creation
    discrete_env = create_car_env(bins=bins)

    single_q_learning_agent = agents.QLearningAgent(
        observation_space=discrete_env.observation_space,
        action_space=discrete_env.action_space,
        exploration_factor=0.5,
    )
    single_q_learning_agent.load("agents/car3.pickle")

    double_q_learning_agent = agents.DoubleQLearningAgent(
        observation_space=discrete_env.observation_space,
        action_space=discrete_env.action_space,
        exploration_factor=0.5,
        weight_initializer=lambda x: np.random.random(size=x) + 10,
    )

    non_discrete_env = create_car_env(discretizate=False, hist=True)
    q_net_agent = csidrl.legacy.agents_neural_legacy.QNetworkAgent(
        non_discrete_env.observation_space,
        non_discrete_env.action_space,
        learning_rate=0.1,
    )

    agent = q_net_agent
    env = non_discrete_env

    # Render base agent
    # precess_episode(env, agent, evalution=True, render=True)

    # Train
    actions = {
        "ipy": lambda agent=agent, env=env, meowtools=meowtools: IPython.embed(),
        "train": lambda: meowtools.train_episode(env, agent),
        "render": lambda: render_episode(agent, bins),
    }

    meowtools.console(actions)

    # Closing
    # agent.save("rl_mountain_car.pickle")
    # env = create_car_env("human", bins, debug=True)
    # precess_episode(env, agent, evalution=True, render=True)


if __name__ == "__main__":
    main()
