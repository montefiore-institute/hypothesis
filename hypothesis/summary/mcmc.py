"""
MCMC Summaries.
"""

import torch
import numpy as np



class Chains:

    def __init__(self, chains):
        self.chains = chains

    def chain(self, index):
        return self.chains[index]

    def size(self):
        return len(self.chains)

    def chain_size(self):
        return self.chains[0].size()

    def mean(self):
        means = []
        for chain in self.chains:
            means.append(chain.mean().view(1, -1))

        return torch.cat(means, dim=0).mean()

    def variance(self):
        variances = []
        for chain in self.chains:
            variances.append(chain.variance().view(1, -1))

        return torch.cat(variances, dim=0).mean()

    def rhat(self):
        n = self.chain_size()
        m = self.size()
        means = []
        for chain in self.chains:
            means.append(chain.mean().view(1, -1))
        chains_mean = torch.cat(means, dim=0).mean()
        W = self.variance()
        B = 0.
        for mean in means:
            B += (mean - chains_mean)
        B /= (n / (m - 1))
        var_theta = (1 - 1/n) * W + (1 / n) * B

        return (var_theta / W).sqrt()


    def gelman_rubin(self):
        return self.rhat()



class Chain:

    def __init__(self, chain,
                 probabilities,
                 acceptances,
                 burnin_chain=None,
                 burnin_probabilities=None,
                 burnin_acceptances=None):
        chain = torch.tensor(chain).squeeze()
        instance_dim = chain[0].dim()
        if instance_dim == 0:
            chain = chain.view(-1, 1)
        probabilities = torch.tensor(probabilities).squeeze()
        acceptances = acceptances
        self._chain = chain
        self._probabilities = probabilities
        self._acceptances = acceptances
        if burnin_chain and burnin_probabilities:
            burnin_chain = torch.tensor(burnin_chain).squeeze()
            if instance_dim == 0:
                burnin_chain = burnin_chain.view(-1, 1)
            burnin_probabilities = torch.tensor(burnin_probabilities).squeeze()
            burnin_acceptances = burnin_acceptances
        self._burnin_chain = burnin_chain
        self._burnin_probabilities = burnin_probabilities
        self._burnin_acceptances = burnin_acceptances

    def has_burnin(self):
        return self._burnin_chain is not None and \
               self._burnin_probabilities is not None

    def mean(self, parameter_index=None, burnin=False):
        result = None
        if burnin:
            if self.has_burnin():
                result = self._burnin_chain[:, parameter_index].mean()
        else:
            result = self._chain[:, parameter_index].mean()

        return result

    def variance(self, burnin=False):
        variance = 0.

        if burnin:
            if self.has_burnin():
                variance = self._burnin_chain.std(dim=0) ** 2
        else:
            variance = self._chain.std(dim=0) ** 2

        return variance

    def size(self):
        return self.iterations()

    def min(self):
        return self._chain.min()

    def max(self):
        return self._chain.max()

    def parameters(self):
        return self._chain[0].view(-1).dim()

    def acceptances(self, burnin=False):
        acceptances = None

        if burnin:
            if self.has_burnin():
                acceptances = self._burnin_acceptances
        else:
            acceptances = self._acceptances

        return acceptances

    def acceptance_ratio(self, burnin=False):
        raise NotImplementedError

    def iterations(self, burnin=False):
        iterations = 0
        if burnin and self.has_burnin():
            iterations = self._burnin_chain.size(0)
        else:
            iterations = self._chain.size(0)

        return iterations

    def chain(self, parameter_index=None, burnin=False):
        chain = None

        if burnin:
            if self.has_burnin():
                chain = self._burnin_chain
        else:
            chain = self._chain
        if chain is not None:
            chain = chain[:, parameter_index].squeeze()

        return chain

    def probabilities(self, parameter_index=None, burnin=False):
        probabilities = None

        if burnin:
            if self.has_burnin():
                probabilities = self._burnin_probabilities
        else:
            probabilities = self._chain
        if probabilities is not None and not parameter_index:
            probabilities = probabilities[:, parameter_index]

        return probabilities.squeeze()

    def autocorrelation(self, lag, parameter_index=None):
        with torch.no_grad():
            num_parameters = self.parameters()
            thetas = self._chain.clone()
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

    def autocorrelation_function(self, max_lag=None, interval=1, parameter_index=None):
        if not max_lag:
            max_lag = self.iterations() - 1
        x = np.arange(0, max_lag + 1, interval)
        y_0 = self.autocorrelation(0, parameter_index)
        y = [self.autocorrelation(tau, parameter_index) / y_0 for tau in x]

        return x, y

    def last(self):
        return self._chain[-1]

    def first(self):
        return self._chain[0]

    def append(self, chain):
        # TODO Appends the specified chain.
        raise NotImplementedError

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
        chain = []
        probabilities = []
        acceptances = []
        for index in range(self.size()):
            u = np.random.uniform()
            if u <= p:
                chain.append(self._chain[index])
                probabilities.append(self._probabilities[index])
                acceptances.append(self._acceptances[index])

        return Chain(chain, probabilities, acceptances)

    def probabilities(self, burnin=False):
        if burnin:
            return self._burnin_probabilities
        else:
            return self._probabilities
