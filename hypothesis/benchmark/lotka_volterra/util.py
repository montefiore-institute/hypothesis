r"""Utilities for the Lotka-Volterra population benchmark.

"""

import torch


@torch.no_grad()
def Prior():
    epsilon = 0.00001
    lower = torch.tensor(4 * [-4]).float()  # In ln-scale
    upper = torch.tensor(4 * [1]).float()  # In ln-scale
    upper += epsilon  # Account for half-open interval

    return Uniform(lower, upper)


class Uniform(torch.distributions.uniform.Uniform):

    def __init__(self, lower, upper):
        lower = lower.to(h.accelerator)
        upper = upper.to(h.accelerator)
        super(Uniform, self).__init__(lower, upper)

    def log_prob(self, sample):
        sample = sample.view(-1, 4)

        return super(Uniform, self).log_prob(sample).sum(dim=1).view(-1, 1)
