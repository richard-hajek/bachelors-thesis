import functools
import pickle

import gym
import numpy as np
import torch
from torch import optim as optim, nn as nn

from csidrl.agents.agents import Agent
from csidrl.legacy.meownets_legacy import QNetwork, train


class QNetworkAgent(Agent):
    def __init__(
        self,
        observation_space,
        action_space,
        learning_rate=0.5,
        discount_factor=0.99,
        exploration_factor=0.5,
        hidden_size=32,
        buffer_size=100_000,
    ):
        assert issubclass(
            type(observation_space), gym.spaces.Box
        ), f"Expected gym.spaces.Box, got {type(observation_space)}"
        assert issubclass(
            type(action_space), gym.spaces.Discrete
        ), f"Expected gym.spaces.Discrete, got {type(action_space)}"

        self.observation_space_n = functools.reduce(
            lambda x, y: x * y, observation_space.shape, 1
        )
        self.qnetwork = QNetwork(self.observation_space_n, action_space.n, hidden_size)
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.action_space = action_space.n
        self.discount_factor = discount_factor
        self.exploration_factor = exploration_factor
        self.replay_buffer = []
        self.buffer_size = buffer_size
        self.batch_size = 32

        self.performance = []

    def action(self, state, evaluation=False):
        state = state.reshape(-1)

        if not evaluation and np.random.uniform() <= self.exploration_factor:
            return np.random.randint(0, self.action_space)

        state = torch.from_numpy(state).float().unsqueeze(0)
        q_values = self.qnetwork(state)
        return q_values.argmax().item()

    def observe(self, state, action, next_state, reward, done):
        state = state.reshape(-1)
        next_state = next_state.reshape(-1)

        self.replay_buffer.append((state, action, next_state, reward, done))
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)

    def update(self, i):
        if i % 32 != 0:
            return

        if len(self.replay_buffer) < self.batch_size:
            return

        states = []
        targets = []

        for i in np.random.choice(
            range(len(self.replay_buffer)), size=self.batch_size, replace=False
        ):
            state, action, next_state, reward, done = self.replay_buffer[i]
            state = torch.Tensor(state)
            next_state = torch.Tensor(next_state)
            q_predicted = self.qnetwork(state)
            q_target = reward

            if not done:
                q_target = q_target + self.discount_factor * torch.max(
                    self.qnetwork(next_state)
                )

            q_predicted[action] = q_target

            states.append(state.detach())
            targets.append(q_predicted.detach())

        train(self.qnetwork, self.optimizer, self.criterion, 10, zip(states, targets))

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load(self, path):
        with open(path, "rb") as f:
            pickled = pickle.load(f)

        self.__dict__.update(pickled.__dict__)

    def on_episode_finish(self):
        pass

    def what_would_agent_do(self, state):
        state = state.reshape(-1)
        state = torch.from_numpy(state).float().unsqueeze(0)
        q_values = self.qnetwork(state)
        return str(q_values)

    def how_to_make_agent(self, state, action):
        state = state.reshape(-1)

        state = torch.tensor(state, requires_grad=True, dtype=torch.float)

        action = torch.from_numpy(action)

        actions = self.qnetwork(state)
        actions.backward(action)

        return state.grad.data

if __name__ == "__main__":
    observation_space = gym.spaces.Box(0, 1, shape=(10,))
    actions_space = gym.spaces.Discrete(n=2)
    agent = QNetworkAgent(observation_space, actions_space)

    state_example = np.ones((10,), dtype=float)
    action_example = np.array([1.0, 0.0])

    print("WWAD")
    print(agent.what_would_agent_do(state_example))

    print("HTMA")
    print(agent.how_to_make_agent(state_example, action_example))