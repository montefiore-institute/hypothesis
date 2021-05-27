r"""Utilities for the Gravitational Wave benchmark.

"""

import hypothesis as h
import torch



@torch.no_grad()
def Prior():
    r"""Returns a prior ``Uniform(10., 80)`` over both masses."""
    epsilon = 0.00001
    lower = torch.tensor([10, 10]).float()
    upper = torch.tensor([80, 80]).float()
    upper += epsilon  # To deal with the half-open interval of the uniform distribution.

    return Uniform(lower, upper)


class Uniform(torch.distributions.uniform.Uniform):

    def __init__(self, lower, upper):
        lower = lower.to(h.accelerator)
        upper = upper.to(h.accelerator)
        super(Uniform, self).__init__(lower, upper)

    def log_prob(self, sample):
        sample = sample.view(-1, 2)
        return super(Uniform, self).log_prob(sample).sum(dim=1).view(-1, 1)
