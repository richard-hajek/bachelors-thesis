import pickle

import gym
import numpy as np
import torch
from torch import optim, nn

from csidrl.agents.agents import Agent
from csidrl.datatypes import can_coerce, type_coerce
from csidrl.deep.meownets import MeowModule, DoubleLinear
import copy



def largest_divisible(n, m):
    return (n // m) * m


class MeowQNetworkAgent(Agent):
    def __init__(
        self,
        observation_space,
        action_space,
        q_network: MeowModule,
        discount_factor=0.99,
        learning_rate=0.89,
        exploration_factor=0.5,
        buffer_size=100_000,
        dl_train_rate=128,
        dl_target_update_rate=512,
        max_dataset_size=10_000,
    ):
        assert issubclass(
            type(observation_space), gym.spaces.Box
        ), f"Expected gym.spaces.Box, got {type(observation_space)}"
        assert issubclass(
            type(action_space), gym.spaces.Discrete
        ), f"Expected gym.spaces.Discrete, got {type(action_space)}"

        assert issubclass(
            type(q_network), MeowModule
        ), f"Expected MeowModule, got {type(q_network)}"

        ok, err = can_coerce(
            observation_space,
            action_space,
            q_network.input_shape(),
            q_network.output_shape(),
            reraise=True,
        )

        if not ok:
            raise RuntimeError(f"Cannot use this network for this space, {err}")

        self.action_space = action_space
        self.observation_space = observation_space

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_factor = exploration_factor
        self.replay_buffer = []
        self.buffer_size = buffer_size
        self.max_dataset_size = max_dataset_size

        self.dl_train_rate = dl_train_rate
        self.dl_target_update_rate = dl_target_update_rate

        self.q_network = q_network
        self.target_q_network = copy.deepcopy(self.q_network)
        self.network_param_type = torch.float
        self.network_input_shape = q_network.input_shape()
        self.network_output_shape = q_network.output_shape()

    def action(self, state, evaluation=False):

        if not evaluation and np.random.uniform() <= self.exploration_factor:
            return np.random.randint(0, self.action_space.n)

        state = type_coerce(state, self.network_input_shape, self.network_param_type)
        q_values = self.q_network(state)
        return q_values.argmax().item()

    def observe(self, state, action, next_state, reward, done):
        state = type_coerce(state, self.network_input_shape, self.network_param_type)
        next_state = type_coerce(
            next_state, self.network_input_shape, self.network_param_type
        )

        state = state.detach()
        next_state = next_state.detach()

        self.replay_buffer.append((state, action, next_state, reward, done))
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)

    def generate_dataset(self):
        dataset = []

        dataset_size = min(self.max_dataset_size, len(self.replay_buffer))
        dataset_size = largest_divisible(dataset_size, self.q_network.batch_size)

        replay_memories = np.random.choice(
            range(len(self.replay_buffer)),
            size=dataset_size,
            replace=False,
        )

        for i in replay_memories:
            state, action, next_state, reward, done = self.replay_buffer[i]

            # Double Q-Learning modification
            q_predicted = self.q_network(state)
            q_target = reward

            if not done:
                q_val = self.q_network(next_state)
                q_argmax = torch.argmax(q_val)
                next_q_values = self.target_q_network(next_state)
                next_q_val = next_q_values[q_argmax]

                q_target = q_target + self.learning_rate * (reward + next_q_val * self.discount_factor - q_val[action])

            q_predicted[action] = q_target

            dataset.append((state.detach(), q_predicted.detach()))

        return dataset

    def update(self, step_number: int):

        if step_number % self.dl_train_rate != 0:
            return

        if len(self.replay_buffer) < self.q_network.batch_size:
            return

        self.DL_train()

        update_target = step_number % self.dl_target_update_rate == 0

        if update_target:
            self.update_target()

    def DL_train(self, epochs=10, tb_writer=None, verbose=False):
        dataset = self.generate_dataset()
        self.last_dataset = dataset
        self.q_network.train(epochs, dataset, tb_writer, verbose)

    def update_target(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    def load(self, path):
        with open(path, "rb") as f:
            pickled = pickle.load(f)

        self.__dict__.update(pickled.__dict__)
        return path

    def dump_memories(self, path, state_repr_func=None, action_repr_func=None):
        import pandas as pd
        df = pd.DataFrame(data=self.replay_buffer, columns=["state","action", "next_state", "reward", "done"])

        if state_repr_func:
            df["state"] = df["state"].apply(state_repr_func)

        if action_repr_func:
            df["action"] = df["action"].apply(action_repr_func)

        df.to_csv(path)

    @staticmethod
    def load_new(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def on_episode_finish(self):
        pass

    def what_would_agent_do(self, state):
        state = type_coerce(state, self.network_input_shape, self.network_param_type)
        q_values = self.q_network(state)
        return str(q_values)

    def how_to_make_agent(self, state, action):
        state = type_coerce(state, self.network_input_shape, self.network_param_type)
        action = type_coerce(action, self.network_output_shape, self.network_param_type)

        state = state.requires_grad_(True)
        actions = self.q_network(state)
        actions.backward(action)

        return state.grad.data


if __name__ == "__main__":
    observation_space = gym.spaces.Box(0, 1, shape=(10,))
    actions_space = gym.spaces.Discrete(n=2)
    state_example = np.ones((10,), dtype=float)
    action_example = np.array([1.0, 0.0])

    q_network = MeowModule(
        DoubleLinear,
        input_shape=(10,),
        output_shape=(2,),
        optimizer=optim.Adam,
        loss=nn.MSELoss,
        batch_size=32,
        lr=0.5,
    )

    new_agent = MeowQNetworkAgent(observation_space, actions_space, q_network)

    print("Meow WWAD")
    print(new_agent.what_would_agent_do(state_example))

    print("Meow HTMA")
    print(new_agent.how_to_make_agent(state_example, action_example))

    dataset = [(state_example, action_example)]

    for i, (s, a) in enumerate(dataset * 32):
        new_agent.observe(s, a, s, 1, False)
        new_agent.update(i)

