from typing import Tuple

import gym
import torch


def can_coerce(
    observation_space: gym.spaces.Box,
    action_space: gym.spaces.Discrete,
    input_shape: Tuple[int, ...],
    output_shape: Tuple[int, ...],
    reraise=False,
) -> Tuple[bool, str]:
    observation_space_example = torch.ones(size=observation_space.shape)
    action_space_example = torch.ones(size=(action_space.n,))

    try:
        observation_space_example.reshape(input_shape)
    except RuntimeError as e:
        if reraise:
            raise

        return False, f"Cannot reshape input shape due to  {str(e)}"

    try:
        action_space_example.reshape(output_shape)
    except RuntimeError as e:
        if reraise:
            raise

        return False, f"Cannot reshape output shape due to {str(e)}"

    return True, ""


def type_coerce(tensor, target_shape, target_type) -> torch.Tensor:
    if type(tensor) != torch.Tensor:
        tensor = torch.tensor(tensor, dtype=target_type)

    tensor = tensor.type(target_type)
    tensor = tensor.reshape(target_shape)
    return tensor


def reduce_shape_tuple(shape) -> int:
    import functools
    return functools.reduce(lambda x, y: x * y, shape, 1)