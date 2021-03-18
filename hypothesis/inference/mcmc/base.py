r"""Base for Markov Chain Monte Carlo samplers.

"""

import torch

from hypothesis.engine import Procedure


class BaseMarkovChainMonteCarlo(Procedure):
    r""""""

    def __init__(self, prior):
        super(BaseMarkovChainMonteCarlo, self).__init__()
        self._prior = prior

    def _register_events(self):
        pass  # No events to register.

    def _step(self, theta, observations):
        raise NotImplementedError

    def reset(self):
        pass  # Nothing to reset.

    @torch.no_grad()
    def sample(self, observations, num_samples, num_burnin=0, initial_sample=None):
        r""""""
        samples = []
        # Check if an initial sample has been set.
        if initial_sample is None:
            initial_sample = self._prior.sample()
        # Reset the MCMC chain.
        self.reset()
        # Set the current sample.
        sample = initial_sample.view(1, -1)
        # Sample the burnin chain.
        for _ in range(num_burnin):
            sample = self._step(sample, observations).view(1, -1)
        # Sample the actual chain.
        for _ in range(num_samples):
            sample = self._step(sample, observations)
            samples.append(sample.view(1, -1))

        return torch.cat(samples, dim=0)
