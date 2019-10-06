r"""Summary objects and statistics for Markov chain Monte Carlo methods."""

import numpy as np
import torch
import warnings



class Chain:
    r"""Summary of a Markov chain produced by an MCMC sampler."""

    def __init__(self, samples, acceptance_probabilities, acceptances):
        self.acceptance_probabilities = acceptance_probabilities
        self.acceptances = acceptances
        self.samples = samples
        self.shape = samples.shape

    def mean(self, parameter_index=None):
        with torch.no_grad():
            mean = self.samples[:, parameter_index].mean(dim=0).squeeze()

        return mean

    def std(self, parameter_index=None):
        with torch.no_grad():
            std = self.samples[:, parameter_index].std(dim=0).squeeze()

        return std

    def variance(self, parameter_index=None):
        with torch.no_grad():
            variance = self.std(parameter_index) ** 2

        return variance

    def monte_carlo_error(self):
        with torch.no_grad():
            mc_error = (self.variance() / self.effective_size()).sqrt()

        return mc_error

    def size(self):
        return len(self.samples)

    def min(self):
        return self.samples.min(dim=0)

    def max(self):
        return self.samples.max(dim=0)

    def dimensionality(self):
        return self.samples.shape[1:][0]

    def autocorrelation(self, lag):
        return self.autocorrelations()[lag]

    def autocorrelations(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            samples = self.samples.numpy()
            samples = np.atleast_1d(samples)
            axis = 0
            m = [slice(None), ] * len(samples.shape)
            n = samples.shape[axis]
            f = np.fft.fft(samples - np.mean(samples, axis=axis), n=2 * n, axis=axis)
            m[axis] = slice(0, n)
            samples = np.fft.ifft(f * np.conjugate(f), axis=axis)[m].real
            m[axis] = 0
            acf = samples / samples[m]

        return torch.from_numpy(acf).float()

    def integrated_autocorrelation(self, max_lag=None):
        autocorrelations = self.autocorrelations()
        integrated_autocorrelation = 0.
        if max_lag is None:
            max_lag = self.size()
        a_0 = autocorrelations[0]
        for index in range(max_lag):
            integrated_autocorrelation += autocorrelations[index]

        return integrated_autocorrelation

    def integrated_autocorrelations(self, interval=1, max_lag=None):
        autocorrelations = self.autocorrelations()
        integrated_autocorrelation = 0.
        integrated_autocorrelations = []
        if max_lag is None:
            max_lag = self.size()
        a_0 = autocorrelations[0]
        for index in range(max_lag):
            integrated_autocorrelation += autocorrelations[index]
            if index % interval == 0:
                integrated_autocorrelations.append(integrated_autocorrelation)

        return integrated_autocorrelations

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
