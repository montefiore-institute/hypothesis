r"""Summary objects and statistics for Markov chain Monte Carlo methods."""

import numpy as np
import torch



class Chain:
    r"""Summary of a Markov chain produced by an MCMC sampler."""

    def __init__(self, samples, acceptance_probabilities, acceptances):
        self.acceptance_probabilities = acceptance_probabilities
        self.acceptances = acceptances
        self.samples = samples
        self.shape = samples.shape

    def mean(self, parameter_index=None):
        return self.samples[:, parameter_index].mean(dim=0)

    def std(self, parameter_index=None):
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

    def dimensionality(self):
        return self.samples.shape[1:]

    def autocorrelation(self, lag, parameter_index=None):
        with torch.no_grad():
            thetas = self.chain.clone()
            sample_mean = self.mean(parameter_index)
            if lag > 0:
                padding = torch.zeros(lag, num_parameters)
                lagged_thetas = thetas[lag:, parameter_index].clone()
                lagged_thetas -= sample_mean
                padded_thetas = torch.cat([lagged_thetas, padding], dim=0)
            else:
                padded_thetas = thetas
            thetas -= sample_mean
            rhos = thetas * padded_thetas
            rho = rhos.sum(dim=0).squeeze()
            rho *= (1. / (self.size() - lag))
        del thetas
        del padded_thetas
        del rhos

        return rho

    def integrated_autocorrelation(self, interval=1, M=0):
        int_tau = 0.
        if not M:
            M = self.size() - 1
        c_0 = self.autocorrelation(0)
        for index in range(M):
            int_tau += self.autocorrelation(index) / c_0

        return int_tau

    def effective_size(self):
        y_0 = self.autocorrelation(0)
        M = 0
        for lag in range(self.size()):
            y = self.autocorrelation(lag)
            p = y / y_0
            if p <= 0:
                M = lag - 1
                break
        tau = self.integrated_autocorrelation(M)
        effective_size = (self.size() / tau)

        return int(abs(effective_size))

    def efficiency(self):
        return self.effective_size() / self.size()

    def thin(self):
        raise NotImplementedError

    def __getitem__(self, pattern):
        return self.samples[pattern]
