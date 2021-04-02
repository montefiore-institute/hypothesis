r"""Ammortized Approximate Likelihood Ratio Metropolis Hastings

https://arxiv.org/abs/1903.04057
"""

import hypothesis as h
import numpy as np
import torch

from hypothesis.inference.mcmc import BaseMarkovChainMonteCarlo


class AALRMetropolisHastings(BaseMarkovChainMonteCarlo):

    def __init__(self, prior, ratio_estimator, proposal):
        super(AALRMetropolisHastings, self).__init__(prior)
        self._denominator = None
        self._r = ratio_estimator
        self._proposal = proposal

    @torch.no_grad()
    def _compute_ratio(self, input, outputs):
        num_observations = outputs.shape[0]
        inputs = input.repeat(num_observations, 1)
        inputs = inputs.to(h.accelerator)
        _, log_ratios = self.ratio_estimator(inputs=inputs, outputs=outputs)

        return log_ratios.sum().cpu()

    @torch.no_grad()
    def _step(self, theta, observations):
        theta_next = self._proposal.sample(theta)
        lnl_theta_next = self._compute_ratio(theta_next, observations)
        numerator = self.prior.log_prob(theta_next) + lnl_theta_next
        if self._denominator is None:
            lnl_theta = self._compute_ratio(theta, observations)
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
