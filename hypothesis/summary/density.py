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



class KernelDensity(Density):

    def __init__(self, samples, bandwidth="auto", kernel=kde_normal):
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



def kde_estimate_bandwidth(samples, method="scott"):
    methods = {
        "scott": kde_estimate_bandwidth_scott}

    return methods[method](samples)



def kde_estimate_bandwidth_scott(samples):
    d = samples.shape[1]
    n = samples.shape[0]

    return n ** (-1./ (d + 4))



def kde_normal(samples, x, bandwidth):
    with torch.no_grad():
        normal = Normal(x, bandwidth)
        n = samples.shape[0]
        k_h = normal.log_prob(samples - x).exp().sum(dim=0)
        p_x = (1 / n) * k_h

    return p_x.item()


def kde_multivariate_normal(samples, x, bandwidth):
    with torch.no_grad():
        x = torch.tensor(x).float().view(1, -1)
        d = x.shape[1]
        normal = MultivariateNormal(x, bandwidth * torch.eye(d))
        k_h = normal.log_prob(samples - x).exp().sum(dim=0)
        p_x = (1 / n) * k_h

    return p_x.squeeze()
