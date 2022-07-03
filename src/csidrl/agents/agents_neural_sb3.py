from csidrl.agents.agents import Agent
import torch
from torch import nn
import torch.optim as optim
import gym
import numpy as np
from stable_baselines3 import DQN

from csidrl.datatypes import type_coerce


class DQNAgentWrapper(Agent):
    def __init__(self, agent):
        self.agent = agent

    def action(self, state, evaluation=False):
        action, _ = self.agent.predict(state)
        return action

    def observe(self, state, action, next_state, reward, done):
        self.agent.collect_rollout(state, action, reward, next_state, done)

    def save(self, path):
        self.agent.save(path)

    def load(self, path):
        self.agent = DQN.load(path)

    def on_episode_finish(self):
        pass

    def what_would_agent_do(self, state):
        state = torch.tensor(state)
        state = torch.unsqueeze(state, dim=0)
        state = state.to("cuda")
        thinking = self.agent.q_net(state)
        return thinking