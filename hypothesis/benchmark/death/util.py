r"""Utilities for the Death Model benchmark.

"""

import torch

from torch.distributions.binomial import Binomial
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform



def Prior():
    r"""Prior over the infection rate."""
    return Normal(1, 1)


def PriorExperiment():
    r"""Prior over the experimental design space (measurement time)."""
    return Uniform(0.1, 10.0)


def Truth():
    return torch.tensor([0.15, 0.05])


def log_likelihood(theta, x):
    raise NotImplemented
