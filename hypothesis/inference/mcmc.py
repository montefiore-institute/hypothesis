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
        self.denominator = None

    def _step(self, observations, theta):
        accepted = False

        theta_next = self.transition.sample(theta)
        lnl_theta_next = self.log_likelihood(theta_next, observations)
        numerator = self.prior.log_prob(theta_next) + lnl_theta_next
        if self.denominator is None:
            lnl_theta = self.log_likelihood(theta, observations)
            self.denominator = self.prior.log_prob(theta) + lnl_theta
        acceptance_ratio = (numerator - self.denominator)
        if not self.transition.is_symmetrical():
            raise NotImplementedError
        acceptance_probability = min([1, acceptance_ratio.exp().item()])
        u = np.random.uniform()
        if u <= acceptance_probability:
            accepted = True
            theta = theta_next

        return theta, acceptance_probability, accepted



class AALRMetropolisHastings(MarkovChainMonteCarlo):
    r"""Ammortized Approximate Likelihood Ratio Metropolis Hastings

    """

    def __init__(self, prior, ratio_estimator, transition):
        super(RatioEstimatorMetropolisHastings, self).__init__()
        self.prior = prior
        self.ratio_estimator = ratio_estimator
        self.transition = transition
        self.denominator = None

    def _compute_ratio(self, observations, theta):
        num_observations = len(observations)
        thetas = theta.repeat(num_observations, 1)
        _, log_ratio = self.ratio_estimator(thetas, observations)

        return log_ratio.sum().cpu()

    def _step(self, observations, theta):
        accepted = False

        theta_next = self.transition.sample(theta)
        lnl_theta_next = self._compute_ratio(theta_next, observations)
        numerator = self.prior.log_prob(theta_next) + lnl_theta_next
        if self.denominator is None:
            lnl_theta = self._compute_ratio(theta, observations)
            self.denominator = self.prior.log_prob(theta) + lnl_theta
        acceptance_ratio = (numerator - self.denominator)
        if not self.transition.is_symmetrical():
            raise NotImplementedError
        acceptance_probability = min([1, acceptance_ratio.exp().item()])
        u = np.random.uniform()
        if u <= acceptance_probability:
            accepted = True
            theta = theta_next

        return theta, acceptance_probability, accepted
