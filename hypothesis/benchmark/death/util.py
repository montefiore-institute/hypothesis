r"""Utilities for the Death Model benchmark.

"""

import torch

from torch.distributions.binomial import Binomial
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform



def PriorExperiment():
    r"""Prior over the experimental design space (measurement time)."""
    return Uniform(0., 10.0)


def Truth():
    return torch.tensor([1.])


def log_likelihood(theta, x):
    raise NotImplementedError



class Prior:

    def __init__(self):
        self.normal = Normal(1, 1)
        self.uniform = Uniform(0, 10)

    def _sample(self):
        sample = None

        neg_infinity = float("-inf")
        while sample is None:
            candidate = self.normal.sample()
            if self.uniform.log_prob(candidate) > neg_infinity:
                sample = candidate
                break

        return sample

    def sample(self, sample_shape=torch.Size()):
        samples = []

        if len(sample_shape) == 0:
            n = 1
        else:
            n = sample_shape[0]
        for _ in range(n):
            samples.append(self._sample().view(-1, 1))

        return torch.cat(samples, dim=0)

    def log_prob(self, sample):
        raise NotImplementedError
