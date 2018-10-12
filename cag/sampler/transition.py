"""
Transition proposals.
"""

import numpy as np
import torch

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal



class TransitionDistribution:

    def log_prob(self, thetas_next, thetas_current):
        raise NotImplementedError

    def sample(self, thetas, samples=1):
        raise NotImplementedError


class NormalTransitionDistribution(TransitionDistribution):

    def __init__(self, sigma=1.):
        super(NormalTransitionDistribution, self).__init__()
        self._sigma = sigma

    def log_prob(self, thetas_next, thetas_current):
        normal = Normal(thetas_current, self._sigma)

        return normal.log_prob(thetas_next)

    def sample(self, thetas, samples=1):
        with torch.no_grad():
            size = torch.Size([samples])
            samples = Normal(thetas.squeeze(), self._sigma).sample(size)

        return samples.view(thetas.size(0), 1, samples)


class MultivariateNormalTransitionDistribution(TransitionDistribution):

    def __init__(self, sigma):
        self._dimensionality = sigma.size(0)
        self._sigma = sigma

    def log_prob(self, thetas_next, thetas_current):
        normal = MultivariateNormal(thetas_current, self._sigma)

        return normal.log_prob(thetas_next)

    def sample(self, thetas, samples=1):
        x = []

        with torch.no_grad():
            thetas = thetas.view(-1, 3)
            size = torch.Size([samples])
            for theta in thetas:
                samples = MultivariateNormal(theta.view(-1), self._sigma).sample(size).view(-1, self._dimensionality, samples)
                x.append(samples)
            x = torch.cat(x, dim=0)

        return x
