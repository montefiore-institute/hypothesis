r"""Utilities for the Weinberg benchmark.

"""

import torch


@torch.no_grad()
def Prior():
    r"""Returns a uniform prior between 0 and 1 over the infection and
    recovery rate (encoded in this order).

    """
    lower = torch.tensor([0, 0]).float()
    upper = torch.tensor([1, 1]).float()

    return torch.distributions.uniform.Uniform(lower, upper)


@torch.no_grad()
def PriorExperiment():
    r"""Returns a Prior ``Uniform(0.0, 10.0)`` over
    the experimental design space (measurement time).

    By default, the simulator will draw samples from
    this distribution to draw experimental configurations.
    """
    return torch.distributions.uniform.Uniform(0.0, 10.0)


@torch.no_grad()
def Truth():
    r"""Returns the true infection and recovery rate of this
    benchmark problem: ``(0.8, 0.2)``.

    """
    return torch.tensor([0.8, 0.2])
