r"""Utilities for the spatial SIR benchmark.

"""

import torch

from hypothesis.exception import IntractableException



def Prior():
    r"""Prior over the infection and recovery rates."""
    lower = torch.tensor([0, 0]).float()
    upper = torch.tensor([1, 1]).float()

    return Uniform(lower, upper)


def PriorExperiment():
    r"""Prior over the experimental design space (measurement time)."""
    return Uniform(0.1, 10.0)


def Truth():
    return torch.tensor([0.15, 0.05])


def log_likelihood(theta, x):
    raise IntractableException



class Uniform(torch.distributions.uniform.Uniform):

    r"""Used to initialize the prior over the experimental design space."""
    def __init__(self, lower, upper):
        super(Uniform, self).__init__(lower, upper)

    def log_prob(self, sample):
        return super(Uniform, self).log_prob(sample).mean()
