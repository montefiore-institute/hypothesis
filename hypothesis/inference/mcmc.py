"""
Markov Chain Monte Carlo methods for inference.
"""

import numpy as np
import torch

from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal

from hypothesis.engine import event
from hypothesis.inference import Method
from hypothesis.inference import SimulatorMethod
from hypothesis.util import sample
from hypothesis.util import epsilon



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


class MarkovChainMonteCarlo(Method):

    KEY_INITIAL_THETA = "theta_0"
    KEY_NUM_SAMPLES = "samples"
    KEY_BURN_IN_STEPS = "burnin_steps"

    def __init__(self):
        super(MarkovChainMonteCarlo, self).__init__()

    def step(self, observations, theta):
        raise NotImplementedError

    def run_chain(self, theta_0, observations, num_samples):
        thetas = []
        probabilities = []
        acceptances = []

        for sample_index in range(num_samples):
            theta_0, acceptance, accepted = self.step(observations, theta_0)
            thetas.append(theta_0.squeeze())
            probabilities.append(acceptance)
            acceptances.append(accepted)

        return thetas, probabilities, acceptances

    def procedure(self, observations, **kwargs):
        burnin_thetas = None
        burnin_probabilities = None
        burnin_acceptances = None

        # Initialize the sampling procedure.
        theta_0 = kwargs[self.KEY_INITIAL_THETA]
        num_samples = int(kwargs[self.KEY_NUM_SAMPLES])
        if self.KEY_BURN_IN_STEPS in kwargs.keys():
            burnin_steps = int(kwargs[self.KEY_BURN_IN_STEPS])
            if burnin_steps > 0:
                # Start the burnin procedure.
                burnin_thetas, burnin_probabilities, burnin_acceptances = self.run_chain(theta_0, observations, burnin_steps)
                # Take the last theta as the initial starting point.
                theta_0 = burnin_thetas[-1]
        # Start sampling form the MH chain.
        thetas, probabilities, acceptances = self.run_chain(theta_0, observations, num_samples)
        chain = Chain(
            thetas, probabilities, acceptances,
            burnin_thetas, burnin_probabilities, burnin_acceptances)

        return chain



class HamiltonianMonteCarlo(MarkovChainMonteCarlo):

    def __init__(self, log_likelihood,
                 leapfrog_steps,
                 leapfrog_stepsize):
        super(HamiltonianMonteCarlo, self).__init__()
        self.log_likelihood = log_likelihood
        self.leapfrog_steps = leapfrog_steps
        self.leapfrog_stepsize = leapfrog_stepsize
        self.momentum = None

    def _allocate_momentum(self, observations):
        if observations[0].dim() == 0:
            mean = torch.tensor(0).float()
            std = torch.tensor(1).float()
            self.momentum = Normal(mean, std)
        else:
            size = len(observations[0])
            mean = torch.zeros(size)
            std = torch.eye(size)
            self.momentum = MultivariateNormal(mean, std)

    def U(self, theta, observations):
        return -self.log_likelihood(theta, observations)

    def dU(self, theta, observations):
        theta = theta.detach()
        theta.requires_grad = True
        ln_likelihood = self.U(theta, observations)
        ln_likelihood.backward()
        gradient = theta.grad.detach()

        return gradient

    def K(self, momentum):
        return (momentum ** 2 / 2).sum()

    def dK(self, momentum):
        momentum = momentum.detach()
        momentum.requires_grad = True
        energy = self.K(momentum)
        energy.backward()
        gradient = momentum.grad.detach()

        return gradient

    def step(self, observations, theta):
        accepted = False
        momentum = self.momentum.rsample()

        momentum_next = momentum
        theta_next = theta
        for step in range(self.leapfrog_steps):
            momentum_next = momentum_next - (self.leapfrog_stepsize / 2.) * self.dU(theta_next, observations)
            theta_next = theta_next + self.leapfrog_stepsize * self.dK(momentum_next)
            momentum_next = momentum_next - (self.leapfrog_stepsize / 2.) * self.dU(theta_next, observations)
        rho = (self.U(theta, observations) - self.U(theta_next, observations) - self.K(momentum_next) + self.K(momentum)).exp()
        acceptance = min([1, rho])
        u = np.random.uniform()
        if u <= acceptance:
            accepted = False
            theta = theta_next

        return theta, acceptance, accepted

    def procedure(self, observations, **kwargs):
        # Allocate the momentum distribution.
        self._allocate_momentum(observations)
        # Initiate the MCMC procedure.
        result = super().procedure(observations, **kwargs)

        return result



class MetropolisHastings(MarkovChainMonteCarlo):

    def __init__(self, log_likelihood,
                 transition):
        super(MetropolisHastings, self).__init__()
        self.log_likelihood = log_likelihood
        self.transition = transition

    def step(self, observations, theta):
        accepted = False

        with torch.no_grad():
            theta_next = self.transition.sample(theta).squeeze()
            likelihood_current = self.log_likelihood(theta, observations)
            likelihood_next = self.log_likelihood(theta_next, observations)
            lr = likelihood_next - likelihood_current
            if not self.transition.is_symmetric():
                t_theta_next = self.transition.log_prob(theta, theta_next).exp()
                t_theta = self.transition.log_prob(theta_next, theta).exp()
                p *= (t_theta_next / (t_theta + epsilon))
            else:
                p = 1
            acceptance = min([1, lr.exp() * p])
            u = np.random.uniform()
            if u <= acceptance:
                accepted = True
                theta = theta_next

        return theta, acceptance, accepted
