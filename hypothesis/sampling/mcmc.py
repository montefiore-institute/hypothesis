"""
Markov Chain Monte Carlo methods for inference.
"""

import copy
import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal

from hypothesis.engine import event
from hypothesis.sampling import Method
from hypothesis.util import sample
from hypothesis.util import epsilon



class Chain:

    def __init__(self, chain, probabilities,
                 burnin_chain=None, burnin_probabilities=None):
        self._chain = chain
        self._probabilities = probabilities
        self._burnin_chain = burnin_chain
        self._burnin_probabilities = burnin_probabilities

    def has_burnin(self):
        return self._burnin_chain is not None and self._burnin_probabilities is not None

    def num_parameters(self):
        return self._chain[0].view(-1).size(0)

    def iterations(self):
        return len(self._chain)

    def effective_size(self):
        raise NotImplementedError

    def burnin_iterations(self):
        iterations = 0
        if self._burnin_chain:
            iterations = len(self._burnin_chain)

        return iterations

    def chain(self, parameter_index=None):
        if not parameter_index:
            return self._chain
        chain = []
        for sample in self._chain:
            chain.append(sample[parameter_index])

        return chain

    def autocorrelation(self, lag):
        raise NotImplementedError

    def probabilities(self):
        return self._probabilities

    def burnin_chain(self, parameter_index=None):
        if not parameter_index:
            return self._burnin_chain
        chain = []
        for sample in self._burnin_chain:
            chain.append(sample[parameter_index])

        return chain

    def burnin_probabilities(self):
        return self._burnin_probabilities



class MetropolisHastings(Method):

    KEY_INITIAL_X = "x_0"
    KEY_NUM_SAMPLES = "samples"
    KEY_BURN_IN_STEPS = "burnin_steps"

    def __init__(self, log_likelihood,
                 transition):
        super(MetropolisHastings, self).__init__()
        self.log_likelihood = log_likelihood
        self.transition = transition

    def step(self, x):
        with torch.no_grad():
            x_next = self.transition.sample(x).squeeze()
            likelihood_current = self.log_likelihood(x)
            likelihood_next = self.log_likelihood(x_next)
            lr = likelihood_next - likelihood_current
            if not self.transition.is_symmetric():
                t_x_next = self.transition.log_prob(x, x_next).exp()
                t_x = self.transition.log_prob(x_next, x).exp()
                p *= (t_x_next / (t_x + epsilon))
            else:
                p = 1
            acceptance = min([1, lr.exp() * p])
            u = np.random.uniform()
            if u <= acceptance:
                x = x_next

        return x, acceptance

    def run_chain(self, x_0, num_samples):
        samples = []
        probabilities = []

        for sample_index in range(num_samples):
            x_0, acceptance = self.step(x_0)
            samples.append(x_0.squeeze())
            probabilities.append(acceptance)

        return samples, probabilities

    def procedure(self, **kwargs):
        burnin_samples = None
        burnin_probabilities = None

        # Initialize the sampling procedure.
        x_0 = kwargs[self.KEY_INITIAL_X]
        num_samples = int(kwargs[self.KEY_NUM_SAMPLES])
        if self.KEY_BURN_IN_STEPS in kwargs.keys():
            burnin_steps = int(kwargs[self.KEY_BURN_IN_STEPS])
            if burnin_steps > 0:
                # Start the burnin procedure.
                burnin_samples, burnin_probabilities = self.run_chain(x_0, burnin_steps)
                # Take the last x as the initial starting point.
                x_0 = burnin_samples[-1]
        # Start sampling form the MH chain.
        samples, probabilities = self.run_chain(x_0, num_samples)
        chain = Chain(
            samples, probabilities,
            burnin_samples, burnin_probabilities)

        return chain



class HamiltonianMonteCarlo(Method):

    KEY_INITIAL_X = "x_0"
    KEY_NUM_SAMPLES = "samples"
    KEY_BURN_IN_STEPS = "burnin_steps"

    def __init__(self, log_likelihood,
                 leapfrog_steps, leapfrog_stepsize):
        super(HamiltonianMonteCarlo, self).__init__()
        self.log_likelihood = log_likelihood
        self.leapfrog_steps = leapfrog_steps
        self.leapfrog_stepsize = leapfrog_stepsize

    def U(self, x):
        return -self.log_likelihood(x)

    def dU(self, x):
        x_var = torch.autograd.Variable(x, requires_grad=True)
        out = self.U(x_var)
        out.backward()
        gradient = x_var.grad
        x_var.detach()
        return gradient

    def K(self, momentum):
        #return ((torch.t(momentum) * momentum) / 2).sum()
        # unit mass assumed
        return ((momentum * momentum) / 2).sum()

    def dK(self, momentum):
        momentum_var = torch.autograd.Variable(momentum, requires_grad=True)
        out = self.K(momentum_var)
        out.backward()
        gradient = momentum_var.grad
        momentum_var.detach()
        return gradient

    def step(self, x):
        if(x.dim() == 0):
            dimensionality = 1
        else:
            dimensionality = len(x)

        if(dimensionality == 1):
            momentum = Normal(torch.tensor([0.0]), torch.tensor([1.0])).rsample()
        else:
            momentum = MultivariateNormal(torch.zeros(dimensionality),
                                              torch.eye(dimensionality)).rsample()

        for l in range(self.leapfrog_steps):
            momentum_next = momentum - self.leapfrog_stepsize/2.0*self.dU(x)
            x_next = x + self.leapfrog_stepsize*self.dK(momentum_next)
            momentum_next = momentum_next - self.leapfrog_stepsize/2.0*self.dU(x_next)

        ro = (-self.U(x_next) + self.U(x) - \
             self.K(momentum_next) + self.K(momentum)).exp()

        p_accept = min(1, ro)
        if(p_accept >= np.random.uniform()): x = x_next

        return x, p_accept

    def run_chain(self, x_0, num_samples):
        samples = []
        probabilities = []

        for sample_index in range(num_samples):
            x_0, acceptance = self.step(x_0)
            samples.append(x_0.squeeze())
            probabilities.append(acceptance)

        return samples, probabilities

    def procedure(self, **kwargs):
        burnin_samples = None
        burnin_probabilities = None

        # Initialize the sampling procedure.
        x_0 = kwargs[self.KEY_INITIAL_X]
        num_samples = int(kwargs[self.KEY_NUM_SAMPLES])
        if self.KEY_BURN_IN_STEPS in kwargs.keys():
            burnin_steps = int(kwargs[self.KEY_BURN_IN_STEPS])
            if burnin_steps > 0:
                # Start the burnin procedure.
                burnin_samples, burnin_probabilities = self.run_chain(x_0, burnin_steps)
                # Take the last x as the initial starting point.
                x_0 = burnin_samples[-1]
        # Start sampling form the MH chain.
        samples, probabilities = self.run_chain(x_0, num_samples)
        chain = Chain(
            samples, probabilities,
            burnin_samples, burnin_probabilities)

        return chain
