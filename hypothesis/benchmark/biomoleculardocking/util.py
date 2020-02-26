r"""Utilities for the Biomolecular Docking benchmark.

"""

import torch

from hypothesis.exception import IntractableException
from torch.distributions.beta import Beta
from torch.distributions.normal import Normal



def Truth():
    raise NotImplementedError


def log_likelihood(theta, x):
    raise IntractableException



class Prior:

    def __init__(self):
        self.r_bottom = Beta(4, 96)
        self.r_ee50 = Normal(-50, 15 ** 2)
        self.r_slope = Normal(-0.15, 0.1 ** 2)
        self.r_top = Beta(25, 75)

    def sample(self, sample_shape=torch.Size()):
        bottom_samples = self.r_bottom.sample(sample_shape).view(-1, 1)
        ee50_samples = self.r_ee50.sample(sample_shape).view(-1, 1)
        slope_samples = self.r_slope.sample(sample_shape).view(-1, 1)
        top_samples = self.r_top.sample(sample_shape).view(-1, 1)
        samples = torch.cat([
            bottom_samples,
            ee50_samples,
            slope_samples,
            top_samples],
            dim=1)

        return samples

    def log_prob(self, sample):
        raise IntractableException



class PriorExperiment(torch.distributions.uniform.Uniform):

    def __init__(self):
        lower = torch.ones(100) * -75
        upper = torch.zeros(100)

    def log_prob(self, sample):
        return super(Uniform, self).log_prob(sample).mean()
