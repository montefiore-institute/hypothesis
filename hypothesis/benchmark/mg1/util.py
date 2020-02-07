r"""Utilities for the M/G/1 benchmark.

"""

import torch
import torch.distributions.uniform

from hypothesis.exception import IntractableException


def allocate_prior():
    lower = torch.tensor([0, 0, 0]).float()
    upper = torch.tensor([10, 10, 1/3]).float()
    return Uniform(lower, upper)


def allocate_truth():
    return torch.tensor([1, 5, .2]).float()


def log_likelihood(theta, x):
    raise IntractableException


class Uniform(torch.distributions.uniform.Uniform):

    def __init__(self, lower, upper):
        super(Uniform, self).__init__(lower, upper)

    def log_prob(self, sample):
        return super(Uniform, self).log_prob(sample).mean()
