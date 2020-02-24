r"""Utilities for the Weinberg benchmark.

"""

import torch

from hypothesis.exception import IntractableException



def Prior():
    r"""Prior over the infection and recovery rates."""
    return Uniform(0, 0, 0.5, 0.5)


def PriorExperiment():
    r"""Prior over the experimental design space (measurement time)."""
    return Uniform(0, 3)


def Truth():
    return torch.tensor([0.15, 0.05])


def log_likelihood(theta, x):
    raise IntractableException



class Uniform(torch.distributions.uniform.Uniform):

    r"""Used to initialize the prior over the experimental design space."""
    def __init__(self, lower, upper):
        super(Uniform, self).__init__(float(lower), float(upper))


    r"""Used to initialize the prior over the model parameters."""
    def __init__(self, beta_lower, gamma_lower, beta_upper, gamma_upper):
        super(Uniform, self).__init__(torch.tensor([float(beta_lower), float(gamma_lower)]),
                                      torch.tensor([float(beta_upper), float(gamma_upper)]))


    def log_prob(self, sample):
        return super(Uniform, self).log_prob(sample).mean()
