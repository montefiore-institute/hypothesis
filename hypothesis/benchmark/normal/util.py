r"""Utilities for the normal benchmark.

"""

import torch

from torch.distributions.normal import Normal



def Prior():
    return Uniform(-10, 10)


def PriorExperiment():
    return Uniform(-10, 10)


def Truth():
    return torch.tensor([0]).float()


def log_likelihood(theta, x):
    return Normal(theta, 1).log_prob(x)



class Uniform(torch.distributions.uniform.Uniform):

    def __init__(self, lower, upper):
        super(Uniform, self).__init__(lower, upper)

    def log_prob(self, sample):
        return super(Uniform, self).log_prob(sample).mean()
