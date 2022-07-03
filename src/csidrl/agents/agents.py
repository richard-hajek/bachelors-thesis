import abc
import pickle

import gym
import numpy as np


def reduce_shape_tuple(shape) -> int:
    import functools
    return functools.reduce(lambda x, y: x * y, shape, 1)


class Agent(abc.ABC):
    @abc.abstractmethod
    def action(self, state, evaluation=False):
        pass

    @abc.abstractmethod
    def observe(self, state, action, next_state, reward, done):
        pass

    @abc.abstractmethod
    def save(self, path):
        pass

    @abc.abstractmethod
    def load(self, path):
        pass

    @abc.abstractmethod
    def on_episode_finish(self):
        pass

    def what_would_agent_do(self, state):
        return "-"


class RandomAgent(Agent):
    def __init__(self, actionspace_n):
        self.actionspace_n = actionspace_n

    def action(self, state, evaluation=False):
        k = np.random.randint(0, self.actionspace_n)
        action_space = np.zeros(shape=(self.actionspace_n,))
        action_space[k] = 1
        return action_space

    def observe(self, state, action, next_state, reward, done):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass

    def on_episode_finish(self):
        pass


class QLearningAgent(Agent):
    def __init__(
        self,
        observation_space,
        action_space,
        learning_rate=0.5,
        discount_factor=0.99,
        exploration_factor=0.5,
        weight_initializer=None,
    ):
        assert issubclass(
            type(observation_space), gym.spaces.Discrete
        ), f"Expected gym.spaces.Discrete, got {type(observation_space)}"
        assert issubclass(
            type(action_space), gym.spaces.Discrete
        ), f"Expected gym.spaces.Discrete, got {type(action_space)}"

        self.states = observation_space.n
        self.actions = action_space.n

        if weight_initializer is None:
            self.Q = np.random.random(size=(self.states, self.actions))
        else:
            self.Q = weight_initializer((self.states, self.actions))
        self.performance = []

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_factor = exploration_factor
        self.QCount = np.zeros((self.states, self.actions))
        self.Qorg = np.array(self.Q)

    def action(self, state, evaluation=False):
        if not evaluation and np.random.uniform() <= self.exploration_factor:
            return np.random.randint(0, self.actions)
        return np.argmax(self.Q[state])

    def observe(self, state, action, next_state, reward, done):
        td_error = (
            reward
            + self.discount_factor * np.max(self.Q[next_state])
            - self.Q[state, action]
        )
        self.Q[state, action] += self.learning_rate * td_error
        self.performance.append(td_error)
        self.QCount[state, action] += 1

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load(self, path):
        with open(path, "rb") as f:
            pickled = pickle.load(f)

        self.__dict__.update(pickled.__dict__)

    def on_episode_finish(self):
        pass


class DoubleQLearningAgent(Agent):
    def __init__(
        self,
        observation_space,
        action_space,
        learning_rate=0.5,
        discount_factor=0.99,
        exploration_factor=0.5,
        weight_initializer=None,
    ):
        assert issubclass(
            type(observation_space), gym.spaces.Discrete
        ), f"Expected gym.spaces.Discrete, got {type(observation_space)}"
        assert issubclass(
            type(action_space), gym.spaces.Discrete
        ), f"Expected gym.spaces.Discrete, got {type(action_space)}"

        self.states = observation_space.n
        self.actions = action_space.n

        if weight_initializer is None:
            self.Q1 = np.random.random(size=(self.states, self.actions))
            self.Q2 = np.array(self.Q1, copy=True)
        else:
            self.Q1 = weight_initializer((self.states, self.actions))
            self.Q2 = np.array(self.Q1, copy=True)

        self.performance = []

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_factor = exploration_factor
        self.QCount = np.zeros((self.states, self.actions))
        self.Qorg = np.array(self.Q1)

    def action(self, state, evaluation=False):
        if not evaluation and np.random.uniform() <= self.exploration_factor:
            return np.random.randint(0, self.actions)
        return np.argmax(0.5 * self.Q1[state] + 0.5 * self.Q2[state])

    def observe(self, state, action, next_state, reward, done):
        if np.random.randint(0, 2) == 0:
            QA = self.Q1
            QB = self.Q2
        else:
            QA = self.Q2
            QB = self.Q1

        td_error = (
            reward
            + self.discount_factor * QB[next_state, np.argmax(QA[next_state])]
            - QA[state, action]
        )

        QA[state, action] += self.learning_rate * td_error

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load(self, path):
        with open(path, "rb") as f:
            pickled = pickle.load(f)

        self.__dict__.update(pickled.__dict__)

    def on_episode_finish(self):
        pass
