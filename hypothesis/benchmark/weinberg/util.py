r"""Utilities for the Weinberg benchmark.

"""

import torch

from hypothesis.exception import IntractableException



def Prior():
    r"""Prior over the Fermi constant."""
    return Uniform(0.5, 1.5)


def PriorExperiment():
    r"""Prior over the experimental design space (the beam-energy)."""
    return Uniform(40, 50)


def Truth():
    return torch.tensor([1]).float()


def log_likelihood(theta, x):
    raise IntractableException



class Uniform(torch.distributions.uniform.Uniform):

    def __init__(self, lower, upper):
        super(Uniform, self).__init__(lower, upper)

    def log_prob(self, sample):
        return super(Uniform, self).log_prob(sample).mean()
