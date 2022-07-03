import satnet.models
from gym.vector.utils import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th # who the fuck thought this was a good idea
from torch import nn

from csidrl.environments.hydraulic_rotations import COMPACT_BITS_DIR, COMPACT_BITS_KIND


class SATNetFeaturesExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        self.flatten = nn.Flatten()

        # Compute shape by doing one forward pass
        with th.no_grad():
            state = th.as_tensor(observation_space.sample()[None]).float()
            self.channels = state.shape[-1] - 1
            pipes = state[:, :, :, 1:]
            n_flatten = self.flatten(pipes).shape[1]

        self.satnet = satnet.models.SATNet(n=n_flatten, m=3*3, max_iter=1)
        self.linear = nn.Linear(n_flatten * 3, features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:

        x_water = observations[:, :, :, 0]
        x_water = x_water.reshape(*x_water.shape, 1)
        x_water = x_water.expand((-1, -1, -1, self.channels))
        x_water[:, :, :, 1:COMPACT_BITS_KIND] = 1
        x_water = x_water.int()

        x_pipes = observations[:, :, :, 1:]

        x_water = self.flatten(x_water)
        x_pipes = self.flatten(x_pipes)
        x = self.satnet(x_pipes, x_water)
        x = th.cat((x, x_pipes, x_water), dim=1)
        x = self.linear(x)
        return x

