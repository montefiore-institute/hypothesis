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



class MetropolisHastings(MarkovChainMonteCarlo):
    r""""""

    def __init__(self, prior, log_likelihood, transition):
        super(MetropolisHastings, self).__init__(prior)
        self.log_likelihood = log_likelihood
        self.proposal = proposal
        self.transition = transition

    def _step(self, observations, theta):
        accepted = False

        lnl_theta = self.log_likelihood(theta, observations)
        theta_next = self.transition.sample(theta)
        lnl_theta_next = self.log_likelihood(theta_next, observations)
        acceptance_ratio = (
            (self.prior.log_prob(theta_next) + lnl_theta_next)
            -
            (self.prior.log_prob(theta) + lnl_theta))
