import inspect
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from csidrl.legacy.meownets_legacy import train


def safe_kwarg_call(f, kwargs):
    safe_kwargs = {
        k: v for k, v in kwargs.items() if k in inspect.signature(f).parameters
    }
    return f(**safe_kwargs)


def reduce_shape_tuple(shape) -> int:
    import functools
    return functools.reduce(lambda x, y: x * y, shape, 1)


class MeowModule:
    """
    This module adds more runtime information
    Optionally, this module may implement training
    """

    def __init__(
            self,
            nn_module,
            input_shape,
            output_shape,
            optimizer,
            loss,
            batch_size,
            dtype=torch.float,
            device="cpu",
            **kwargs
    ):

        input_shape = (reduce_shape_tuple(input_shape), )
        output_shape = (reduce_shape_tuple(output_shape), )

        self.input_shape_tuple = input_shape
        self.output_shape_tuple = output_shape
        self.batch_size = batch_size
        self.dtype = dtype
        self.device = device

        kwargs["input_neurons"] = input_shape[0]
        kwargs["output_neurons"] = output_shape[0]

        if "lr" in kwargs:
            kwargs["lr"] = kwargs["lr"] / batch_size

        self.nn_module: nn.Module = safe_kwarg_call(nn_module, kwargs)
        self.nn_module.to(dtype=dtype, device=torch.device(device))

        kwargs["params"] = self.nn_module.parameters()

        self.loss: nn.Module = safe_kwarg_call(loss, kwargs)
        self.optimizer: torch.optim.Optimizer = safe_kwarg_call(optimizer, kwargs)

    def sanity_check(self):
        x = torch.zeros(self.input_shape(), device=self.device)
        y = self.nn_module(x)

        assert y.shape == self.output_shape()

    def dump(self):
        torch.set_printoptions(sci_mode=False, precision=2)
        return list(self.nn_module.named_parameters())

    def __call__(self, tensor: torch.Tensor, *args, **kwargs):
        tensor = tensor.to(device=self.device, dtype=self.dtype)
        return self.nn_module(tensor, *args, **kwargs)

    def input_shape(self) -> Tuple:
        return self.input_shape_tuple

    def output_shape(self) -> Tuple:
        return self.output_shape_tuple

    def train(self, epochs, dataset, tb_writer=None, verbose=False):
        assert self.optimizer is not None
        assert self.loss is not None

        torch_dataset = []
        batch_xs = []
        batch_ys = []

        indices = np.random.permutation(len(dataset))

        for i in indices:
            x, y = dataset[i]


            if type(x) != torch.Tensor:
                x = torch.tensor(x)
            if type(y) != torch.Tensor:
                y = torch.tensor(y)

            x = x.to(dtype=self.dtype, device=self.device)
            y = y.to(dtype=self.dtype, device=self.device)

            batch_xs.append(x)
            batch_ys.append(y)

            if len(batch_xs) >= self.batch_size:
                xs = torch.stack(batch_xs)
                ys = torch.stack(batch_ys)

                # If ys end up scalars and therefore only 1D array, loss function complains
                if len(ys.shape) < 2:
                    ys = ys.reshape((-1, 1))

                torch_dataset.append((xs, ys))
                batch_xs.clear()
                batch_ys.clear()

        self.last_dataset = torch_dataset

        self.nn_module.train(True)
        result = train(self.nn_module, self.optimizer, self.loss, epochs, torch_dataset, tb_writer, self.batch_size, verbose=verbose)
        self.nn_module.train(False)

        return result

    def set_optim(self, optim=None, **kwargs):
        kwargs["params"] = self.nn_module.parameters()

        if optim is None:
            optim=type(self.optimizer)

        self.optimizer = safe_kwarg_call(optim, kwargs)

    def __str__(self):
        net = f"{type(self).__name__} - I/O {self.input_shape()} {self.output_shape()}"
        optim = f"optim: {type(self.optimizer).__name__}, params: {self.optimizer}"
        loss = f"loss: {type(self.loss).__name__}, params: {self.loss}"
        return "\n".join([net, optim, loss])

    def load_state_dict(self, *args, **kwargs):
        return self.nn_module.load_state_dict(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.nn_module.state_dict(*args, **kwargs)


class SingleLinear(nn.Module):
    def __init__(self, input_neurons: int, output_neurons: int):
        super().__init__()
        self.fc = nn.Linear(input_neurons, output_neurons)

    def forward(self, x):
        return self.fc(x)


class DoubleLinear(nn.Module):
    def __init__(
            self, input_neurons: int, output_neurons: int, hidden_size: int = 32
    ):
        super().__init__()

        self.fc1 = nn.Linear(input_neurons, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_neurons)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class LinearRelu(nn.Module):
    def __init__(self, input_neurons: int, output_neurons: int):
        super().__init__()
        self.linear = nn.Linear(input_neurons, output_neurons)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.leaky_relu(x)
        return x


class SharpRelu(nn.Module):
    def __init__(self, input_neurons: int, output_neurons: int):
        super().__init__()
        self.linear = nn.Linear(input_neurons, output_neurons)

    def forward(self, x):
        x = self.linear(x)
        x = torch.relu(x)
        return x


class DeepLeakyReluSigmoidOut(nn.Module):
    def __init__(self, input_neurons: int, output_neurons: int):
        super().__init__()

        self.leaky_relu = nn.LeakyReLU()

        self.linear1 = nn.Linear(input_neurons, input_neurons * 2)
        self.linear2 = nn.Linear(input_neurons * 2, input_neurons)
        self.linear3 = nn.Linear(input_neurons, output_neurons)

    def forward(self, x):
        x = self.linear1(x)
        x = self.leaky_relu(x)
        x = self.linear2(x)
        x = self.leaky_relu(x)
        x = self.linear3(x)
        return x

class DeepLeakyRelu(nn.Module):
    def __init__(self, input_neurons: int, output_neurons: int):
        super().__init__()

        self.leaky_relu = nn.LeakyReLU()

        self.linear1 = nn.Linear(input_neurons, input_neurons * 2)
        self.linear2 = nn.Linear(input_neurons * 2, input_neurons)
        self.linear3 = nn.Linear(input_neurons, output_neurons)

    def forward(self, x):
        x = self.linear1(x)
        x = self.leaky_relu(x)
        x = self.linear2(x)
        x = self.leaky_relu(x)
        x = self.linear3(x)
        return x


class DeepLeakyReluSigmoidScaled(nn.Module):
    def __init__(self, input_neurons: int, output_neurons: int):
        super().__init__()
        self.deep_leaky_relu = DeepLeakyRelu(input_neurons, output_neurons)
        self.scale = 100

    def forward(self, x):
        x = self.deep_leaky_relu(x)
        x = torch.sigmoid(x)
        x = x * 100
        return x