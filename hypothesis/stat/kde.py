"""
Kernel Density Estimation
"""

import numpy as np
import torch

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal



def kde_normal(samples, x, bandwidth):
    assert samples.dim() == 2

    with torch.no_grad():
        x = torch.tensor(x).float().view(-1)
        n = samples.shape[0]
        d = samples.shape[1]
        normal = MultivariateNormal(x, bandwidth * torch.eye(d))
        k_h = normal.log_prob(samples - x).sum(dim=0)
        p_x = (1 / n) * k_h

    return p_x.squeeze()


class kernel_density:

    def __init__(self, samples, bandwidth="auto", kernel=kde_normal):
        if bandwidth is "auto":
            self.bandwidth = kde_estimate_bandwidth(samples)
        else:
            self.bandwidth = bandwidth
        self.kernel = kernel
        #self.samples = torch.cat(samples, dim=1)
        self.samples = samples

    def log_prob(self, x):
        return self.kernel(self.samples, x, self.bandwidth)


def kde_estimate_bandwidth(samples, method="scott"):
    methods = {
        "scott": kde_estimate_bandwidth_scott}

    return methods[method](samples)


def kde_estimate_bandwidth(samples):
    assert samples.dim() == 2

    d = samples.shape[1]
    n = samples.shape[0]

    return n ** (-1 / (d + 4))
