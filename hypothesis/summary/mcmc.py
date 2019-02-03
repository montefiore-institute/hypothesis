"""
Summaries and statistics for Markov chain Monte Carlo methods.
"""

import numpy as np
import torch



class Chain:
    r"""
    Summary of a Markov chain from an MCMC sampler.

    Arguments:
       chain (sequence): Sequence of MCMC states
       probabilities (sequence): Sequence of proposal probabilities
       acceptances (sequence): Sequence of accept, reject flags
       burnin_chain (sequence): Sequence of MCMC states during the burnin period
       burnin_probabilities (sequence): Sequence of proposal probabilities during the burnin period
       burnin_acceptances (sequence): Sequence of accept, reject flags during the burnin period
    """

    def __init__(self, chain,
                 probabilities,
                 acceptances,
                 burnin_chain=None,
                 burnin_probabilities=None,
                 burnin_acceptances=None):
        # Initialize the main chain states.
        chain = torch.cat(chain, dim=0).squeeze()
        d = chain[0].dim()
        if d == 0:
            chain = chain.view(-1, 1)
        probabilities = torch.tensor(probabilities).squeeze()
        acceptances = acceptances
        self.chain = chain
        self.probabilities = probabilities
        self.acceptances = acceptances
        # Initialize the burnin chain states
        if burnin_chain:
            burnin_chain = torch.cat(burnin_chain, dim=0).squeeze()
            if d == 0:
                burnin_chain = burnin_chain.view(-1, 1)
            burnin_probabilities = torch.tensor(burnin_probabilities).squeeze()
            burnin_acceptances = burnin_acceptances
        self.burnin_chain = burnin_chain
        self.burnin_probabilities = burnin_probabilities
        self.burnin_acceptances = burnin_acceptances

    def _no_burnin_chain(self):
        return ValueError("No burnin information available.")

    def has_burnin(self):
        return self.burnin_chain is not None

    def mean(self, parameter_index=None, burnin=False):
        chain = self.chain
        if burnin and self.has_burnin():
            chain = self.burnin_chain
        else:
            self._no_burnin_chain()

        return chain[:, parameter_index].mean(dim=0).squeeze()

    def variance(self, parameter_index=None, burnin=False):
        chain = self.chain
        if burnin and self.has_burnin():
            chain = self.burnin_chain
        else:
            self._no_burnin_chain()

        return (chain[:, parameter_index].std(dim=0) ** 2).squeeze()

    def monte_carlo_error(self):
        variance = self.variance()
        effective_sample_size = self.effective_size()

        return (variance / effective_sample_size).sqrt()

    def size(self, burnin=False):
        size = 0
        if burnin and self.has_burnin():
            size = len(self.burnin_chain)
        else:
            size = len(self.chain)

        return size

    def min(self):
        return self.chain.min()

    def max(self):
        return self.chain.max()

    def state_dim(self):
        return self.chain[0].view(-1).dim()

    def acceptances(self, burnin=False):
        acceptances = self.burnin_acceptances
        if burnin and self.has_burnin():
            acceptances = self.burnin_acceptances
        else:
            self._no_burnin_chain()

        return acceptances

    def acceptance_ratio(self):
        raise NotImplementedError

    def get_chain(self, parameter_index=None, burnin=False):
        chain = self.chain
        if burnin and self.has_burnin():
            chain = self.burnin_chain
        else:
            self._no_burnin_chain()

        return chain[:, parameter_index].squeeze().clone()

    def probabilities(self, burnin=False):
        p = self.probabilities
        if burnin and self.has_burnin():
            p = self.burnin_probabilities
        else:
            self._no_burnin_chain()

        return p.squeeze()

    def autocorrelation(self, lag, parameter_index=None):
        with torch.no_grad():
            num_parameters = self.state_dim()
            thetas = self.chain.clone()
            sample_mean = self.mean(parameter_index)
            if lag > 0:
                padding = torch.zeros(lag, num_parameters)
                lagged_thetas = thetas[lag:, parameter_index].view(-1, num_parameters).clone()
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

    def autocorrelation_function(self, parameter_index=None, interval=1, max_lag=None):
        if not max_lag:
            max_lag = self.size() - 1
        x = np.arange(0, max_lag + 1, interval)
        y_0 = self.autocorrelation(lag=0, parameter_index=parameter_index)
        y = [self.autocorrelation(lag=tau, parameter_index=parameter_index) / y_0 for tau in x]

        return x, y

    def integrated_autocorrelation(self, M=None, interval=1):
        int_tau = 0.
        if not M:
            M = self.size() - 1
        c_0 = self.autocorrelation(0)
        for index in range(M):
            int_tau += self.autocorrelation(index) / c_0

        return int_tau

    def efficiency(self):
        return self.effective_size() / self.size()

    def effective_size(self):
        # TODO Support multi-dimensional
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

    def thin(self):
        chain = []
        p = self.efficiency()
        probabilities = []
        acceptances = []
        for index in range(self.size()):
            u = np.random.uniform()
            if u <= p:
                chain.append(self.chain[index])
                probabilities.append(self.probabilities[index])
                acceptances.append(self.acceptances[index])

        return Chain(chain, probabilities, acceptances)
