"""
Markov Chain Monte Carlo.
"""

import numpy as np
import torch

from cag.method import Method
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal



class TransitionDistribution:

    def log_prob(thetas_next, thetas_current):
        raise NotImplementedError

    def sample(thetas, samples=1):
        raise NotImplementedError


class NormalTransitionDistribution(TransitionDistribution):

    def __init__(self, sigma=1.):
        super(NormalTransitionDistribution, self).__init__()
        self._sigma = sigma

    def log_prob(thetas_next, thetas_current):
        normal = Normal(thetas_current, self._sigma)

        return normal.log_prob(thetas_next)

    def sample(thetas, samples=1):
        with torch.no_grad():
            size = torch.Size([samples])
            samples = Normal(thetas.squeeze(), self._sigma).sample(size)

        return samples


class MetropolisHastingsApproximateLikelihoodRatios(Method):

    def __init__(self, simulator,
                 classifier,
                 batch_size=32):
        super(MetropolisHastingsApproximateLikelihoodRatios, self).__init__(simulator)
        self.classifier = classifier
        self.batch_size = batch_size

    def infer(x_o, num_steps=1000):
        raise NotImplementedError
