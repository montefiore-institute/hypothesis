r"""MCMC methods based on Metropolis-Hastings.

"""

import numpy as np
import torch

from hypothesis.inference.mcmc import BaseMarkovChainMonteCarlo


class MetropolisHastings(BaseMarkovChainMonteCarlo):

    def __init__(self, prior, log_likelihood, proposal):
        super(MetropolisHastings, self).__init__(prior)
        self._denominator = None
        self._log_likelihood = log_likelihood
        self._proposal = proposal

    @torch.no_grad()
    def _step(self, theta, observations):
        theta_next = self._proposal.sample(theta)
        lnl_theta_next = self._log_likelihood(theta_next, observations)
        numerator = self.prior.log_prob(theta_next) + lnl_theta_next
        if self._denominator is None:
            lnl_theta = self._log_likelihood(theta, observations)
            self._denominator = self.prior.log_prob(theta) + lnl_theta_next
        acceptance_ratio = numerator - self._denominator
        if not self._proposal.is_symmetrical():
            raise NotImplementedError
        transition_probability = min([1.0, acceptance_ratio.exp().item()])
        u = np.random.uniform()
        if u <= acceptance_probability:
            theta = theta_next
            self._denominator = numerator

        return theta

    def reset(self):
        self._denominator = None
