"""
Utilities for training parameterized classifiers.
"""

import torch
import hypothesis
import os

from hypothesis.train import Trainer
from torch.utils.data import DataLoader



class ParameterizedClassifierTrainer(Trainer):
    r"""Training interface for parameterized classifiers."""

    def __init__(self, dataset, allocate_optimizer, epochs=1, data_workers=2,
                 batch_size=32, checkpoint=None, validate=None,
                 allocate_scheduler=None, criterion=torch.nn.BCELoss(reduction="sum")):
        # Initialize the parent object.
        super(ParameterizedClassifierTrainer, self).__init__(
            dataset, allocate_optimizer, epochs, data_workers,
            batch_size, checkpoint, validate, allocate_scheduler)
        self.epoch_iterations = int(len(dataset) / batch_size / 2)
        self.criterion = criterion
        self.zeros = torch.zeros(self.batch_size, 1)
        self.ones = torch.ones(self.batch_size, 1)

    def dataset_iterations(self):
        return self.epoch_iterations

    def step(self, loader):
        try:
            thetas, x_thetas = next(loader)
            _, x_thetas_hat = next(loader)
            y = self.model(thetas, x_thetas)
            y_hat = self.model(thetas, x_thetas_hat)
            loss = self.criterion(y, self.zeros) + self.criterion(y_hat, self.ones)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        except Exception as e:
            hypothesis.call_hooks(hypothesis.hooks.exception, self, exception=e)
            loss = None

        return loss
