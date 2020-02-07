or"""Likelihood-free Inference By Ratio Estimation.

"""

import numpy as np
import torch

from glmnet import LogitNet
from hypothesis.engine import Procedure



class LFIRE(Procedure, torch.nn.Module):

    def __init__(self, simulator, prior, simulation_batch_size=10000, summary=None):
        super(LFIRE, self).__init__()
        self.prior = prior
        self.simulation_batch_size = int(simulation_batch_size)
        self.simulator = simulator
        self.summary = summary

    def _simulate_marginal_data(self):
        size = torch.Size([self.simulation_batch_size])
        inputs = self.prior.sample(size).view(self.simulation_batch_size, -1)
        outputs = self._simulate(inputs)

        return outputs

    def _simulate_likelihood_data(self, inputs):
        inputs = inputs.repeat(self.simulation_batch_size).view(self.simulation_batch_size, -1)
        outputs = self._simulate(inputs)

        return outputs

    def _simulate(self, inputs):
        outputs = self.simulator(inputs)
        # Check if a custom summary has been applied.
        if self.summary is not None:
            outputs = self.summary(outputs)

        return outputs

    def _register_events(self):
        pass # No events to register.

    def reset(self):
        pass

    def log_ratios(self, inputs, outputs):
        log_ratios = []

        # Simulate data from the marginal model.
        marginal_data = self._simulate_marginal_data()
        # Compute every log likelihood-to-evidence ratio.
        for theta, x in zip(inputs, outputs):
            print(theta, x)
        log_ratios = torch.cat(log_ratios, dim=0)

        return log_ratios
