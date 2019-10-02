r"""Summary objects and statistics for Markov chain Monte Carlo methods."""

import numpy as np
import torch



class Chain:
    r"""Summary of a Markov chain produced by an MCMC sampler."""

    def __init__(self, samples, acceptance_probabilities, acceptances):
        self.acceptance_probabilities = acceptance_probabilities
        self.acceptances = acceptances
        self.samples = samples

    def mean(self, parameter_index=None):
        return self.samples[:, parameter_index].mean(dim=0)

    def std(self, parameter_index=None)
        return self.samples[:, parameter_index].std(dim=0)

    def variance(self, parameter_index=None):
        return self.std(parameter_index) ** 2

    def monte_carlo_error(self):
        return (self.variance() / self.effective_size()).sqrt()

    def size(self):
        return len(self.samples)

    def min(self):
        return self.samples.min(dim=0)

    def max(self):
        return self.samples.max(dim=0)

    def shape(self):
        return self.samples[0].shape

    def autocorrelation(self, lag, parameter_index=None):
        raise NotImplementedError

    def effective_size(self):
        raise NotImplementedError

    def efficiency(self):
        return self.effective_size() / self.size()

    def thin(self):
        raise NotImplementedError
