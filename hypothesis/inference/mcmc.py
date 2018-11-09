"""
Markov Chain Monte Carlo methods for inference.
"""

import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal

from hypothesis.engine import event
from hypothesis.inference import Method
from hypothesis.inference import SimulatorMethod
from hypothesis.util import sample
from hypothesis.util import epsilon



class Chain:

    def __init__(self, chain, probabilities,
                 burnin_chain=None, burnin_probabilities=None):
        chain = torch.tensor(chain).squeeze()
        probabilities = torch.tensor(probabilities).squeeze()
        self._chain = chain
        self._chain_mean = chain.view(len(chain), -1).mean(dim=0)
        self._probabilities = probabilities
        if burnin_chain:
            burnin_chain = torch.tensor(burnin_chain).squeeze()
        if burnin_probabilities:
            burnin_probabilities = torch.tensor(burnin_probabilities)
        self._burnin_chain = burnin_chain
        self._burnin_probabilities = burnin_probabilities

    def has_burnin(self):
        return self._burnin_chain is not None and self._burnin_probabilities is not None

    def chain_mean(self, parameter_index=None):
        return self._chain_mean[parameter_index]

    def chain_min(self):
        return self._chain.min()

    def chain_max(self):
        return self._chain.max()

    def num_parameters(self):
        return self._chain[0].view(-1).size(0)

    def iterations(self):
        return self._chain.size(0)

    def size(self):
        return self.iterations()

    def effective_size(self, interval=None):
        _, y = self.autocorrelation_function(self.iterations(), interval)
        int_autocorrelation = sum(y)
        tau_int = 1 + 2 * (int_autocorrelation)
        n_effective = self.iterations() / tau_int

        return n_effective.round().long().item()

    def burnin_iterations(self):
        iterations = 0
        if self._burnin_chain is not None:
            iterations = self._burnin_chain.size(0)

        return iterations

    def thin(self, size=None):
        if not size:
            size = self.effective_size()
        thinned_chain = self._chain[0:self._chain.size(0):size]
        thinned_probabilities = self._probabilities[0:self._probabilities.size(0):size]

        return Chain(thinned_chain, thinned_probabilities)

    def chain(self, parameter_index=None):
        if parameter_index is None:
            return self._chain

        return self._chain[:, parameter_index]

    def autocorrelation(self, lag, parameter_index=None):
        with torch.no_grad():
            thetas = self._chain
            sample_mean = self._chain_mean[parameter_index]
            num_thetas = len(thetas)
            rho = 0.
            for index in range(num_thetas):
                if index + lag >= num_thetas:
                    break
                rho += (thetas[index][parameter_index] - sample_mean) * (thetas[index + lag][parameter_index] - sample_mean)

        return rho

    def autocorrelation_function(self, max_lag, interval=5, parameter_index=None):
        x = np.arange(0, max_lag + 1, interval)
        y_0 = self.autocorrelation(0, parameter_index)
        y = [self.autocorrelation(tau, parameter_index) / y_0 for tau in x]

        return x, y

    def probabilities(self):
        return self._probabilities

    def burnin_chain(self, parameter_index=None):
        if parameter_index is None:
            return self._burnin_chain

        return self._burnin_chain[:, parameter_index]

    def burnin_probabilities(self):
        return self._burnin_probabilities


class LikelihoodFreeMetropolisHastings(SimulatorMethod):

    KEY_INITIAL_THETA = "theta_0"
    KEY_NUM_SAMPLES = "samples"
    KEY_BURN_IN_STEPS = "burnin_steps"

    def __init__(self, simulator,
                 transition,
                 classifier_allocator,
                 simulator_samples=20000,
                 batch_size=128,
                 epochs=50):
        super(LikelihoodFreeMetropolisHastings, self).__init__(simulator)
        self.transition = transition
        self.classifier_allocator = classifier_allocator
        self._simulator_samples = simulator_samples
        self._batch_size = batch_size
        self._epochs = epochs
        # Add LFHM specific events.
        event.add_event("lfmh_train_start")
        event.add_event("lfmh_train_end")
        event.add_event("lfmh_simulation_start")
        event.add_event("lfmh_simulation_end")
        event.add_event("lfmh_step_start")
        event.add_event("lfmh_step_end")

    def _train_classifier(self, x_theta, x_theta_next):
        self.fire_event(event.lfmh_train_start)
        iterations = int((self._simulator_samples * self._epochs) / self._batch_size)
        classifier = self.classifier_allocator()
        optimizer = torch.optim.Adam(classifier.parameters())
        real = torch.ones(self._batch_size, 1)
        fake = torch.zeros(self._batch_size, 1)
        bce = torch.nn.BCELoss()
        # Start the training procedures.
        for iteration in range(iterations):
            x_fake = sample(x_theta, self._batch_size)
            x_real = sample(x_theta_next, self._batch_size)
            y_real = classifier(x_real)
            y_fake = classifier(x_fake)
            loss = (bce(y_real, real) + bce(y_fake, fake)) / 2.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.fire_event(event.lfmh_train_end)

        return classifier

    def step(self, observations, theta):
        with torch.no_grad():
            theta = theta.unsqueeze(0)
            self.fire_event(event.lfmh_simulation_start)
            theta_next = self.transition.sample(theta).squeeze().unsqueeze(0)
            theta_in = torch.cat([theta] * self._simulator_samples, dim=0)
            theta_next_in = torch.cat([theta_next] * self._simulator_samples, dim=0)
            _, x_theta = self.simulator(theta_in)
            _, x_theta_next = self.simulator(theta_next_in)
            self.fire_event(event.lfmh_simulation_end)
        classifier = self._train_classifier(x_theta, x_theta_next)
        with torch.no_grad():
            s = classifier(observations)
            lr = (s / (1 - s + epsilon)).log().sum().exp()
            if not self.transition.is_symmetric():
                t_theta_next = self.transition.log_prob(theta, theta_next).exp()
                t_theta = self.transition.log_prob(theta_next, theta).exp()
                p *= (t_theta_next / (t_theta + 10e-7))
            else:
                p = 1
            acceptance = min([1, lr.exp() * p])
            u = np.random.uniform()
            if u <= acceptance:
                theta = theta_next

        return theta, acceptance

    def run_chain(self, theta_0, observations, num_samples):
        thetas = []
        probabilities = []

        for sample_index in range(num_samples):
            self.fire_event(event.lfmh_step_start)
            theta_0, acceptance = self.step(observations, theta_0)
            self.fire_event(event.lfmh_step_end)
            thetas.append(theta_0.squeeze())
            probabilities.append(acceptance)

        return thetas, probabilities

    def procedure(self, observations, **kwargs):
        burnin_thetas = None
        burnin_probabilities = None

        # Initialize the sampling procedure.
        theta_0 = kwargs[self.KEY_INITIAL_THETA]
        num_samples = int(kwargs[self.KEY_NUM_SAMPLES])
        if self.KEY_BURN_IN_STEPS in kwargs.keys():
            burnin_steps = int(kwargs[self.KEY_BURN_IN_STEPS])
            if burnin_steps > 0:
                # Start the burnin procedure.
                burnin_thetas, burnin_probabilities = self.run_chain(theta_0, observations, burnin_steps)
                # Take the last theta as the initial starting point.
                theta_0 = thetas[-1]
        # Start sampling form the MH chain.
        thetas, probabilities = self.run_chain(theta_0, observations, num_samples)
        chain = Chain(
            thetas, probabilities,
            burnin_thetas, burnin_probabilities)

        return chain



class MetropolisHastings(Method):

    KEY_INITIAL_THETA = "theta_0"
    KEY_NUM_SAMPLES = "samples"
    KEY_BURN_IN_STEPS = "burnin_steps"

    def __init__(self, log_likelihood,
                 transition):
        super(MetropolisHastings, self).__init__()
        self.log_likelihood = log_likelihood
        self.transition = transition

    def step(self, observations, theta):
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
                theta = theta_next

        return theta, acceptance

    def run_chain(self, theta_0, observations, num_samples):
        thetas = []
        probabilities = []

        for sample_index in range(num_samples):
            theta_0, acceptance = self.step(observations, theta_0)
            thetas.append(theta_0.squeeze())
            probabilities.append(acceptance)

        return thetas, probabilities

    def procedure(self, observations, **kwargs):
        burnin_thetas = None
        burnin_probabilities = None

        # Initialize the sampling procedure.
        theta_0 = kwargs[self.KEY_INITIAL_THETA]
        num_samples = int(kwargs[self.KEY_NUM_SAMPLES])
        if self.KEY_BURN_IN_STEPS in kwargs.keys():
            burnin_steps = int(kwargs[self.KEY_BURN_IN_STEPS])
            if burnin_steps > 0:
                # Start the burnin procedure.
                burnin_thetas, burnin_probabilities = self.run_chain(theta_0, observations, burnin_steps)
                # Take the last theta as the initial starting point.
                theta_0 = burnin_thetas[-1]
        # Start sampling form the MH chain.
        thetas, probabilities = self.run_chain(theta_0, observations, num_samples)
        chain = Chain(
            thetas, probabilities,
            burnin_thetas, burnin_probabilities)

        return chain



class HamiltonianMonteCarlo(Method):

    KEY_INITIAL_THETA = "theta_0"
    KEY_NUM_SAMPLES = "samples"
    KEY_BURN_IN_STEPS = "burnin_steps"

    def __init__(self, log_likelihood,
                 leapfrog_steps, leapfrog_stepsize):
        super(HamiltonianMonteCarlo, self).__init__()
        self.log_likelihood = log_likelihood
        self.leapfrog_steps = leapfrog_steps
        self.leapfrog_stepsize = leapfrog_stepsize

    def U(self, theta, observations):
        return -self.log_likelihood(theta, observations)

    def dU(self, theta, observations):
        theta_var = torch.autograd.Variable(theta, requires_grad=True)
        out = self.U(theta_var, observations)
        out.backward()
        gradient = theta_var.grad
        theta_var.detach()
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

    def step(self, observations, theta):
        if(observations[0].dim() == 0):
            momentum = Normal(torch.tensor([0.0]), torch.tensor([1.0])).rsample()
        else:
            momentum = MultivariateNormal(torch.zeros(len(observations[0])),
                                              torch.eye(len(observations[0]))).rsample()

        for l in range(self.leapfrog_steps):
            momentum_next = momentum - self.leapfrog_stepsize/2.0*self.dU(theta, observations)
            theta_next = theta + self.leapfrog_stepsize*self.dK(momentum_next)
            momentum_next = momentum_next - self.leapfrog_stepsize/2.0*self.dU(theta_next, observations)

        ro = (-self.U(theta_next, observations) + self.U(theta, observations) - \
             self.K(momentum_next) + self.K(momentum)).exp()

        p_accept = min(1, ro)
        if(p_accept >= np.random.uniform()): theta = theta_next

        return theta, p_accept

    def run_chain(self, theta_0, observations, num_samples):
        thetas = []
        probabilities = []

        for sample_index in range(num_samples):
            theta_0, acceptance = self.step(observations, theta_0)
            thetas.append(theta_0.squeeze())
            probabilities.append(acceptance)

        return thetas, probabilities

    def procedure(self, observations, **kwargs):
        burnin_thetas = None
        burnin_probabilities = None

        # Initialize the sampling procedure.
        theta_0 = kwargs[self.KEY_INITIAL_THETA]
        num_samples = int(kwargs[self.KEY_NUM_SAMPLES])
        if self.KEY_BURN_IN_STEPS in kwargs.keys():
            burnin_steps = int(kwargs[self.KEY_BURN_IN_STEPS])
            if burnin_steps > 0:
                # Start the burnin procedure.
                burnin_thetas, burnin_probabilities = self.run_chain(theta_0, observations, burnin_steps)
                # Take the last theta as the initial starting point.
                theta_0 = burnin_thetas[-1]
        # Start sampling form the MH chain.
        thetas, probabilities = self.run_chain(theta_0, observations, num_samples)
        chain = Chain(
            thetas, probabilities,
            burnin_thetas, burnin_probabilities)

        return chain
