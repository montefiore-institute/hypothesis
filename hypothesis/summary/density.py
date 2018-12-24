"""
Density summary.
"""

import torch
import numpy as np

from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal



class Density:

    def log_prob(self, x):
        raise NotImplementedError

    def sample(self, num_samples):
        raise NotImplementedError



class KernelDensity(Density):

    def __init__(self, samples, bandwidth=0.05, kernel="normal"):
        super(KernelDensity, self).__init__()
        if bandwidth is "auto":
            self.bandwidth = kde_estimate_bandwidth(samples)
        else:
            self.bandwidth = bandwidth
        self.kernel = kernel
        self._samples = torch.cat([samples], dim=1)

    def samples(self):
        return self._samples

    def log_prob(self, x):
        raise NotImplementedError

    def sample(self, num_samples):
        raise NotImplementedError



def kde_estimate_bandwidth(samples):
    raise NotImplementedError


def kde_normal(samples, x, bandwidth):
    with torch.no_grad():
        n = samples.dim(0)
        k_h = ((samples - x) / bandwidth).sum(dim=0)
        p_x = (1 / (n * bandwidth)) * k_h

    return p_x
