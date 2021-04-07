r"""Utilities for the Weinberg benchmark.

"""

import hypothesis as h
import torch


@torch.no_grad()
def Prior():
    r"""Returns a uniform prior between 0 and 1 over the infection and
    recovery rate (encoded in this order).

    """
    lower = torch.tensor([0, 0]).float()
    upper = torch.tensor([1, 1]).float()

    return Uniform(lower, upper)


@torch.no_grad()
def PriorExperiment():
    r"""Returns a Prior ``Uniform(0.0, 10.0)`` over
    the experimental design space (measurement time).

    By default, the simulator will draw samples from
    this distribution to draw experimental configurations.
    """
    return torch.distributions.uniform.Uniform(0.1, 10.0)


@torch.no_grad()
def Truth():
    r"""Returns the true infection and recovery rate of this
    benchmark problem: ``(0.8, 0.2)``.

    """
    return torch.tensor([0.8, 0.2])



class Uniform(torch.distributions.uniform.Uniform):

    def __init__(self, lower, upper):
        lower = lower.to(h.accelerator)
        upper = upper.to(h.accelerator)
        super(Uniform, self).__init__(lower, upper)

    def log_prob(self, sample):
        sample = sample.view(-1, 2)
        return super(Uniform, self).log_prob(sample).sum(dim=1)
