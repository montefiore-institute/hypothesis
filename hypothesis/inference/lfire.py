r"""Likelihood-free Inference By Ratio Estimation.

Implementation adapted from `https://github.com/elfi-dev/zoo/blob/master/pylfire/pylfire/methods/lfire.py`
"""

import hypothesis
import numpy as np
import torch

from glmnet import LogitNet
from hypothesis.engine import Procedure



class LFIRE(Procedure, torch.nn.Module):

    def __init__(self, simulator, prior,
            alpha=0.01,
            simulation_batch_size=10000,
            summary=None,
            parallelism=hypothesis.workers,
            approximations=1):
        Procedure.__init__(self)
        torch.nn.Module.__init__(self)
        self.alpha = float(alpha)
        self.approximations = int(approximations)
        self.parallelism = int(parallelism)
        self.prior = prior
        self.simulation_batch_size = int(simulation_batch_size)
        self.simulator = simulator
        self.summary = summary

    def _simulate(self, inputs):
        outputs = self.simulator(inputs)
        # Check if a custom summary has been applied.
        if self.summary is not None:
            outputs = self.summary(outputs)

        return outputs

    def _register_events(self):
        pass # No events to register.

    @torch.no_grad()
    def approximate_log_ratio(self, marginal_data, theta, x, reduce=True):
        likelihood_data = self.simulate_likelihood_data(theta)
        data = torch.cat([likelihood_data, marginal_data], dim=0).numpy()
        ones = torch.ones(self.simulation_batch_size, 1)
        labels = torch.cat([ones, -ones], dim=0).view(-1).numpy()
        x = x.numpy()
        log_ratios = []
        for _ in range(self.approximations):
            model = LogitNet(**{
                "alpha": self.alpha,
                "n_jobs": self.parallelism})
            model.fit(data, labels)
            log_ratio = model.intercept_ + np.sum(np.multiply(model.coef_, x))
            log_ratios.append(torch.tensor(log_ratio).view(1, 1))
            del model # Free memory
        log_ratios = torch.cat(log_ratios, dim=1)
        if reduce:
            result = log_ratios.mean()
        else:
            result = log_ratios

        return result

    def reset(self):
        pass

    def simulate_marginal_data(self):
        size = torch.Size([self.simulation_batch_size])
        inputs = self.prior.sample(size).view(self.simulation_batch_size, -1)
        outputs = self._simulate(inputs)

        return outputs

    def simulate_likelihood_data(self, inputs):
        inputs = inputs.repeat(self.simulation_batch_size).view(self.simulation_batch_size, -1)
        outputs = self._simulate(inputs)

        return outputs

    @torch.no_grad()
    def log_ratios(self, inputs, outputs, marginal_data=None, reduce=True):
        log_ratios = []

        # Check if marginal data has been specified.
        if marginal_data is None:
            # Simulate data from the marginal model.
            marginal_data = self.simulate_marginal_data()
        # Compute every log likelihood-to-evidence ratio.
        for theta, x in zip(inputs, outputs):
            log_ratios.append(self.approximate_log_ratio(marginal_data, theta, x, reduce=reduce).view(-1))
        log_ratios = torch.cat(log_ratios, dim=0)

        return log_ratios
