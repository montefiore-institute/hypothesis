r"""Utilities for the tractable benchmark.

"""

import hypothesis as h
import torch

from torch.distributions.multivariate_normal import MultivariateNormal as Normal


@torch.no_grad()
def Prior():
    lower = -3 * torch.ones(2).float()
    upper = 3 * torch.ones(2).float()

    return Uniform(lower, upper)


@torch.no_grad()
def Truth():
    truth = [0.7, -2.9, -1.0, -0.9]

    return torch.tensor(truth).float()


class Uniform(torch.distributions.uniform.Uniform):

    def __init__(self, lower, upper):
        lower = lower.to(h.accelerator)
        upper = upper.to(h.accelerator)
        super(Uniform, self).__init__(lower, upper)

    def log_prob(self, sample):
        sample = sample.view(-1, 2)

        return super(Uniform, self).log_prob(sample).sum(dim=1).view(-1, 1)
