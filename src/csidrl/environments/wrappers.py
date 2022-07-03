from typing import Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym.core import ActType, ObsType
from matplotlib import animation


class DiscretizedCartPole(gym.ObservationWrapper):
    def __init__(self, env, n_bins=10):
        super().__init__(env)
        self.n_bins = n_bins
        self.observation_space = gym.spaces.Discrete(n_bins**4)
        self.ratios = self.n_bins ** np.arange(0, 4)
        self.upper_bounds = [
            self.env.observation_space.high[0],
            0.5,
            self.env.observation_space.high[2],
            np.radians(50),
        ]
        self.lower_bounds = [
            self.env.observation_space.low[0],
            -0.5,
            self.env.observation_space.low[2],
            -np.radians(50),
        ]
        self.bins = np.linspace(self.lower_bounds, self.upper_bounds, n_bins - 1)

    def observation(self, observation):
        observation_digitezed = np.zeros((4,))

        for i, (obs, bins) in enumerate(zip(observation, self.bins.T)):
            observation_digitezed[i] = np.digitize(obs, bins)

        observation_discrete = np.sum(observation_digitezed * self.ratios)

        assert (
            0 <= observation_discrete < self.observation_space.n
        ), f"Calculated state repr {observation_discrete} when the max is {self.observation_space.n}"
        return int(observation_discrete)


class DiscretizedMountainCar(gym.ObservationWrapper):
    def __init__(self, env, n_bins=10):
        super().__init__(env)
        self.n_bins = n_bins
        self.observation_space = gym.spaces.Discrete(n_bins**2)
        self.ratios = self.n_bins ** np.arange(0, 2)
        self.bins = np.linspace(
            self.env.observation_space.low, self.env.observation_space.high, n_bins - 1
        )

    def observation(self, observation):
        observation_digitezed = np.zeros((2,))

        for i, (obs, bins) in enumerate(zip(observation, self.bins.T)):
            observation_digitezed[i] = np.digitize(obs, bins)

        observation_discrete = np.sum(observation_digitezed * self.ratios)

        assert (
            0 <= observation_discrete < self.observation_space.n
        ), f"Calculated state repr {observation_discrete} when the max is {self.observation_space.n}"
        return int(observation_discrete)


class ActionPrintingCarWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.msgs = ["Left", "Nop", "Right"]

    def action(self, act):
        print(self.msgs[act])
        return act


class ContinousRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super(ContinousRewardWrapper, self).__init__(env)

    def step(self, action, *args, **kwargs):
        next_state, reward, terminated, trunc, info = self.env.step(
            action, *args, **kwargs
        )
        reward += next_state[0]

        if terminated:
            reward += 1000

        return next_state, reward, terminated, trunc, info


class RecordWraper(gym.Wrapper):
    def __init__(self, env):
        super(RecordWraper, self).__init__(env)
        self.actions = []
        self.states = []
        self.rewards = []

    def step(self, action, *args, **kwargs):
        self.actions.append(action)
        next_state, reward, terminated, trunc, info = self.env.step(action)

        self.states.append(next_state)
        self.rewards.append(reward)

        return next_state, reward, terminated, trunc, info


class TrackingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        if not hasattr(env, "get_best_action"):
            raise AttributeError(
                "The wrapped environment does not have the method 'get_best_action'."
            )

        if not hasattr(env, "get_best_score"):
            raise AttributeError(
                "The wrapped env does not have the method get_best_score"
            )

        self.steps_list = []
        self.correct_actions_list = []
        self.steps = 0
        self.correct_actions = 0

    def step(self, action, *args, **kwargs):
        state, reward, done, trunc, info = self.env.step(action, *args, **kwargs)

        self.steps += 1
        self.correct_actions += int(action == self.env.get_best_action(state))

        return state, reward, done, trunc, info

    def reset(self, *args, **kwargs):
        self.steps_list.append(self.steps)
        self.correct_actions_list.append(self.correct_actions)
        self.steps = 0
        self.correct_actions = 0
        return self.env.reset(*args, **kwargs)

    def get_results(self):
        return self.steps_list, self.correct_actions_list

    def report(self, target_file="log.csv"):
        print(f"Best possible score: {self.env.get_best_score()}")
        df = pd.DataFrame(
            data={
                "steps": self.steps_list,
                "correct_actions": self.correct_actions_list,
            }
        )
        df.to_csv(target_file)
        print(df)


class GifRenderer(gym.ObservationWrapper):
    def __init__(self, env, filename):
        self.env = env
        self.frames = []
        self.filename = filename

    def reset(self, *args, **kwargs):
        self.save_frames_as_gif(self.frames, filename=self.filename)
        self.frames = []
        return self.env.reset(*args, **kwargs)

    def step(self, action, *args, **kwargs):
        observation, reward, done, trunc, info = self.env.step(action, *args, **kwargs)
        self.frames.append(self.env.render())

        if done or trunc:
            self.save_frames_as_gif(self.frames, filename=self.filename)
            self.frames = []

        return observation, reward, done, trunc, info

    @staticmethod
    def save_frames_as_gif(frames, path="./", filename="gym_animation.gif"):
        if len(frames) == 0:
            return

        plt.figure(
            figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72
        )
        patch = plt.imshow(frames[0])
        plt.axis("off")

        def animate(i):
            patch.set_data(frames[i])

        try:
            anim = animation.FuncAnimation(
                plt.gcf(), animate, frames=len(frames), interval=50
            )
            print("Saving...")
            anim.save(path + filename, writer="imagemagick", fps=60)
            print("Saved")
        except KeyboardInterrupt as e:
            pass


class HistogramWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observations = []

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.observations.append(observation)
        return observation, info

    def step(self, action):
        observation, reward, done, trunc, info = self.env.step(action)
        self.observations.append(observation)
        return observation, reward, done, trunc, info

    def create_histogram(self, feature_index=None):
        observations = np.array(self.observations)
        num_features = observations.shape[1]
        if feature_index is None:
            for i in range(num_features):
                plt.hist(observations[:, i], bins=10)
                plt.title(f"Feature {i}")
                plt.show()
        else:
            if feature_index >= num_features:
                raise ValueError(
                    f"feature_index must be less than the number of features ({num_features})"
                )
            plt.hist(observations[:, feature_index], bins=10)
            plt.title(f"Feature {feature_index}")
            plt.show()


class BoxidizerWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        obs: gym.spaces.Discrete = env.observation_space
        self.observation_space = gym.spaces.Box(low=0, high=obs.n, shape=(1, ))


class DelayedReward(gym.Wrapper):
    def __init__(self, env, num_episodes):
        super().__init__(env)
        self.num_episodes = num_episodes
        self.episode_rewards = []

    def step(self, action):
        observation, reward, done, trunc, info = self.env.step(action)
        done = done or trunc
        self.episode_rewards.append(reward)

        if done:
            if len(self.episode_rewards) >= self.num_episodes:
                total_reward = sum(self.episode_rewards)
                self.episode_rewards = []
                return observation, total_reward, True, False, info
            else:
                observation, info = self.env.reset()
                done = False

        return observation, 0, done, trunc, info


class CompatibilityWrapper(gym.Wrapper):
    def __int__(self, env):
        self.env = env

    def reset(self, **kwargs) -> Tuple[ObsType, dict]:
        obs, info = self.env.reset(**kwargs)
        return obs

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        o, r, d, t, i = self.env.step(action)
        return o, r, d or t, {}


class FlattenObservations(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.utils.flatten_space(env.observation_space)

    def reset(self, **kwargs) -> Tuple[ObsType, dict]:
        obs, info = self.env.reset()
        obs = obs.reshape((-1,))
        return obs, info

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        o, r, d, t, i = self.env.step(action)
        return o.reshape((-1,)), r, d, t, i


gym.envs.register(
    id="MountainCar1000-v0",
    entry_point="gym.envs.classic_control:MountainCarEnv",
    max_episode_steps=1000,
    reward_threshold=-110.0,
)
