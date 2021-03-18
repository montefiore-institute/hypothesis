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
        pass
