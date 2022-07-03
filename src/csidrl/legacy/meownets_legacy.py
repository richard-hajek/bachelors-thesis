from typing import Tuple

import torch
from torch import nn as nn
import numpy as np

class QNetwork(nn.Module):
    def __init__(
        self, observation_space: int, action_space: int, hidden_size: int = 32
    ):
        super(QNetwork, self).__init__()

        self.input_shape_tuple = (observation_space,)
        self.output_shape_tuple = (action_space,)

        self.fc1 = nn.Linear(observation_space, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_space)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def input_shape(self) -> Tuple:
        return self.input_shape_tuple

    def output_shape(self) -> Tuple:
        return self.output_shape_tuple


def stop_training(epoch_mean_losses, acceptable_deviation=0.001):

    if len(epoch_mean_losses) >= 6 and np.mean(epoch_mean_losses[-6:]) <= acceptable_deviation:
        return True

    if len(epoch_mean_losses) >= 12 and np.abs(np.mean(epoch_mean_losses[-6:]) - np.mean(epoch_mean_losses[-12:])) < acceptable_deviation:
        return True

    if len(epoch_mean_losses) > 16 and np.mean(epoch_mean_losses[-8:]) > np.mean(epoch_mean_losses[-16:-8]):
        return True

    return False


from tqdm import tqdm


def train(model, optimizer, criterion, num_epochs, dataset, tb_writer=None, batch_size=1, verbose=False):
    tb_x = 1

    losses = []
    last_epoch_loss = None
    last_loss = None

    progress_bar = None
    if verbose:
        progress_bar = tqdm(total=num_epochs)

    for epoch_index in range(num_epochs):

        epoch_avg_loss = []

        for i, (inputs, labels) in enumerate(dataset):
            # Reset me thinks
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            last_loss = loss.item() / batch_size

            if verbose:
                progress_bar.set_description(f'batch {i}, last_epoch_loss {last_epoch_loss}, loss {last_loss:.04f}')

            epoch_avg_loss.append(last_loss)

        if tb_writer:
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            tb_x += 1

        losses.append(np.mean(epoch_avg_loss))
        last_epoch_loss = losses[-1]

        if stop_training(losses):
            break

        if verbose:
            progress_bar.update(1)

    # print(f"Epoch {epoch}: Loss = {loss.item()}")
    return losses
