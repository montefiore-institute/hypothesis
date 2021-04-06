r"""Utilities for the M/G/1 benchmark.

"""

import hypothesis as h
import torch


@torch.no_grad()
def Prior():
    r"""Returns a uniform prior between ``(0, 0, 0)`` and
    `(10, 10, 1/3)`. """
    lower = torch.tensor([0.0, 0.0, 0.0])
    upper = torch.tensor([10.0, 10.0, 1 / 3])

    return Uniform(lower, upper)


@torch.no_grad()
def Truth():
    r"""Returns the true queuing model parameters: ``(1, 5, 0.2)``.

    """
    return torch.tensor([1.0, 5.0, 0.2])


class Uniform(torch.distributions.uniform.Uniform):

    def __init__(self, lower, upper):
        lower = lower.to(h.accelerator)
        upper = upper.to(h.accelerator)
        super(Uniform, self).__init__(lower, upper)

    def log_prob(self, sample):
        sample = sample.view(-1, 3)
        return super(Uniform, self).log_prob(sample).sum(dim=1)
