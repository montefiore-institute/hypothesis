r"""Markov chain Monte Carlo methods for inference.
"""

import hypothesis
import numpy as np
import torch

from hypothesis.summary.mcmc import Chain
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal



class MarkovChainMonteCarlo:
    r""""""

    def __init__(self, prior):
        super(MarkovChainMonteCarlo, self).__init__()
        self.prior = prior

    def _step(self, observations, theta):
        raise NotImplementedError

    def sample(self, observations, theta, num_samples):
        r""""""
        acceptance_probabilities = []
        acceptances = []
        samples = []
        for sample_index in range(num_samples):
            theta, acceptance_probability, acceptance = self._step(observations, theta)
            samples.append(theta.view(1, -1))
            acceptance_probabilities.append(acceptance_probability)
            acceptances.append(acceptance)
        chain = Chain(samples, acceptance_probabilities, acceptances)

        return chain
