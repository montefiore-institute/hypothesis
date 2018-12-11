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
from hypothesis.summary.mcmc import Chain
from hypothesis.summary.mcmc import Chains
from hypothesis.util import epsilon
from hypothesis.util import sample



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

        if(thetas[0].size(0) == 1):
            chain = Chain(
                thetas, probabilities, acceptances,
                burnin_thetas, burnin_probabilities, burnin_acceptances)
        else:
            chain = Chains(thetas)

        return chain



class Hamiltonian(MarkovChainMonteCarlo):

    def __init__(self, log_likelihood,
                 leapfrog_steps,
                 leapfrog_stepsize):
        super(Hamiltonian, self).__init__()
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

    def likelihood_ratio(self, observations, theta_next, theta):
        likelihood_current = self.log_likelihood(theta, observations)
        likelihood_next = self.log_likelihood(theta_next, observations)
        lr = likelihood_next - likelihood_current

        return lr

    def step(self, observations, theta):
        accepted = False

        with torch.no_grad():
            theta_next = self.transition.sample(theta).squeeze()
            lr = self.likelihood_ratio(observations, theta_next, theta)
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



class RatioMetropolisHastings(MarkovChainMonteCarlo):

    def __init__(self, ratio,
                 transition):
        super(RatioMetropolisHastings, self).__init__()
        self.ratio = ratio
        self.transition = transition

    def step(self, observations, theta):
        accepted = False

        theta_next = self.transition.sample(theta).squeeze().detach()

        lr = self.ratio(observations, theta_next, theta)
        if not self.transition.is_symmetric():
            t_theta_next = self.transition.log_prob(theta, theta_next).exp()
            t_theta = self.transition.log_prob(theta_next, theta).exp()
            p *= (t_theta_next / (t_theta + epsilon))
        else:
            p = 1
        acceptance = min([1, lr * p])
        u = np.random.uniform()
        if u <= acceptance:
            accepted = True
            theta = theta_next

        return theta, acceptance, accepted



class LikelihoodFreeRatioHamiltonian(MarkovChainMonteCarlo):

    def __init__(self, classifier,
                 leapfrog_steps,
                 leapfrog_stepsize):
        super(LikelihoodFreeRatioHamiltonian, self).__init__()
        self.classifier = classifier
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
        raise NotImplementedError

    def dU(self, theta, observations):
        raise NotImplementedError

    def K(self, momentum):
        return ((momentum ** 2) / 2).sum()

    def dK(self, momentum):
        momentum.requires_grad = True
        energy = self.K(momentum)
        energy.backward()
        gradient = momentum.grad.detach()
        momentum.requires_grad = False

        return gradient

    def step(self, observations, theta):
        # TODO Implement.
        raise NotImplementedError
        accepted = False
        momentum = self.momentum.rsample()

        momentum_next = momentum
        theta_next = theta
        rho = 0.
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
