"""
Markov chain Monte Carlo methods for inference.
"""

import hypothesis
import numpy as np
import torch

from hypothesis.inference import Method
from hypothesis.summary.mcmc import Chain
from hypothesis.util import epsilon
from hypothesis.util import sample

from hypothesis.inference.approximate_likelihood_ratio import log_likelihood_ratio
from hypothesis.inference.approximate_likelihood_ratio import marginal_ratio
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal



class MarkovChainMonteCarlo(Method):
    r"""General interface for Markov chain Monte Carlo (MCMC) procedures.


    .. note: The `infer` procedures takes 4 arguments.
        observations: tensor containing all observations.
        theta_0: initial parameter of the Markov chain.
        samples: number of MCMC samples.
        burnin_samples: number of burnin-samples of the Markov chain.
    """

    KEY_INITIAL_THETA = "theta_0"
    KEY_SAMPLES = "samples"
    KEY_BURNIN_SAMPLES = "burnin_samples"

    def __init__(self):
        super(MarkovChainMonteCarlo, self).__init__()

    def _initialize(self, observations, **kwargs):
        pass # Nothing to do here.

    def step(self, observations, theta):
        raise NotImplementedError

    def sample(self, observations, theta_0, num_samples):
        samples = []
        probabilities = []
        acceptances = []

        theta = theta_0
        for sample_index in range(num_samples):
            hypothesis.call_hooks(hypothesis.hooks.pre_step, self)
            theta, probability, accepted = self.step(observations, theta)
            theta = theta.view(1, -1)
            hypothesis.call_hooks(hypothesis.hooks.post_step, self, theta=theta, probability=probability, accepted=accepted)
            samples.append(theta.detach())
            probabilities.append(probability)
            acceptances.append(accepted)

        return samples, probabilities, acceptances

    def infer(self, observations, **kwargs):
        burnin_samples = None
        burnin_probabilities = None
        burnin_acceptances = None

        # Initialize the sampler.
        self._initialize(observations, **kwargs)
        # Fetch the procedure parameters.
        theta_0 = kwargs[self.KEY_INITIAL_THETA]
        num_samples = int(kwargs[self.KEY_SAMPLES])
        # Check if a burnin-chain needs to be sampled.
        if self.KEY_BURNIN_SAMPLES in kwargs.keys():
            burnin_num_samples = int(kwargs[self.KEY_BURNIN_SAMPLES])
            b_samples, b_probabilities, b_acceptances = self.sample(observations, theta_0, burnin_num_samples)
            burnin_samples = b_samples
            burnin_probabilities = b_probabilities
            burnin_acceptances = b_acceptances
            theta_0 = burnin_samples[-1]
        # Sample the main chain.
        hypothesis.call_hooks(hypothesis.hooks.pre_inference, self)
        samples, probabilities, acceptances = self.sample(observations, theta_0, num_samples)
        hypothesis.call_hooks(hypothesis.hooks.post_inference, self, samples=samples)
        # Allocate the chain summary.
        chain = Chain(samples, probabilities, acceptances,
                      burnin_samples, burnin_probabilities, burnin_acceptances)

        return chain



class RatioMetropolisHastings(MarkovChainMonteCarlo):
    r"""Metropolis-Hastings MCMC sampler in which you define a custom likelihood-ratio function.

    Arguments:
        ratio (Lambda): custom likelihood-ratio function.
        transition (Transition): transition distribution.
    """

    def __init__(self, ratio, transition):
        super(RatioMetropolisHastings, self).__init__()
        self.likelihood_ratio = ratio
        self.transition = transition

    def step(self, observations, theta):
        accepted = False

        theta = theta.squeeze().detach()
        theta_next = self.transition.sample(theta).squeeze().detach()
        lr = self.likelihood_ratio(observations, theta, theta_next).detach()
        if not self.transition.is_symmetric():
            t_next = self.transition.log_prob(theta, theta_next).exp()
            t_current = self.transition.log_prob(theta_next, theta).exp()
            p = (t_next / (t_current + epsilon)).item()
        else:
            p = 1
        probability = min([1, lr * p])
        u = np.random.uniform()
        if u <= probability:
            accepted = True
            theta = theta_next

        return theta, probability, accepted



class ClassifierMetropolisHastings(RatioMetropolisHastings):
    r"""Likelihood-free Classifier Metropolis Hastings MCMC

    In this setting, a pretrained parameterized classifier can be trained to
    perform Metropolis-Hastings MCMC sampling.

    Arguments:
        classifier (torch.nn.Module): a pretrained parameterized classifier with inputs (theta, x_theta).
        transition (Transition): a transition distribution.
    """

    def __init__(self, classifier, transition):
        # Define the ratio functions using the specified classifier.
        def likelihood_ratio(observations, theta, theta_next):
            return classifier.log_likelihood_ratio(observations, theta, theta_next).exp()

        # Initialize the parent with the ratio method.
        super(ClassifierMetropolisHastings, self).__init__(likelihood_ratio, transition)



class MetropolisHastings(RatioMetropolisHastings):
    r"""Metropolis Hastings MCMC sampler

    Arguments:
        log_likelihood (lambda): function that measures the log-likelihood of the observations with a given theta.
        transition (Transtion): transition distribution proposing parameters.

    .. note::
        Possible optimizations possible with the way the likelihood-ratio is obtained.
    """

    def __init__(self, log_likelihood, transition):
        # Define the ratio function in terms of the log-likelihood.
        def likelihood_ratio(observations, theta, theta_next):
            likelihood_current = log_likelihood(observations, theta)
            likelihood_next = log_likelihood(observations, theta_next)
            lr = likelihood_next - likelihood_current

            return lr.exp()
        # Initialize the parent with the ratio-method.
        super(MetropolisHastings, self).__init__(likelihood_ratio, transition)



class Hamiltonian(MarkovChainMonteCarlo):
    r"""Hamiltonian MCMC sampler
    """

    def __init__(self, log_likelihood, momentum=None,
                 leapfrog_steps=50,
                 leapfrog_stepsize=.01):
        super(Hamiltonian, self).__init__()
        self.log_likelihood = log_likelihood
        self.leapfrog_steps = int(leapfrog_steps)
        self.leapfrog_stepsize = float(leapfrog_stepsize)
        self.momentum = momentum

    def _initialize(self, observations, **kwargs):
        # Initialize the momentum distribution.
        self._allocate_momentum(observations)

    def _allocate_momentum(self, observations):
        if self.momentum is None:
            dimensions = observations[0].dim()
            if dimensions == 0:
                momentum = Normal(0, 1)
            else:
                dimensions = len(observations[0])
                loc = torch.zeros(dimensions)
                scale = torch.eye(dimensions)
                momentum = MultivariateNormal(loc, scale)
            self.momentum = momentum

    def U(self, observations, theta):
        return -self.log_likelihood(observations, theta)

    def dU(self, observations, theta):
        theta = theta.detach()
        theta.requires_grad = True
        l = self.U(observations, theta)
        l.backward()
        gradient = theta.grad.detach()

        return gradient

    def K(self, momentum):
        return (.5 * momentum ** 2).sum()

    def dK(self, momentum):
        momentum = momentum.detach()
        momentum.requires_grad = True
        energy = self.K(momentum)
        energy.backward()
        gradient = momentum.grad.detach()

        return gradient

    def step(self, observations, theta):
        theta = theta.squeeze().detach()
        momentum = self.momentum.sample()
        accepted = False

        m_next = momentum.clone()
        t_next = theta.clone()
        # Leapfrog integration
        eta = self.leapfrog_stepsize / 2
        for step in range(self.leapfrog_steps):
            m_next -= eta * self.dU(observations, t_next)
            t_next += self.leapfrog_stepsize * self.dK(m_next)
            m_next -= eta * self.dU(observations, t_next)
        U_initial = self.U(observations, theta)
        U_final = self.U(observations, t_next)
        K_initial = self.K(momentum)
        K_final = self.K(m_next)
        rho = (U_initial - U_final + K_initial - K_final).exp().item()
        probability = min([1, rho])
        u = np.random.uniform()
        if u <= probability:
            accepted = True
            theta = t_next

        return theta, probability, accepted



class ClassifierHamiltonian(MarkovChainMonteCarlo):
    r"""Likelihood-free Hamiltonian MCMC sampler
    """

    def __init__(self, classifier, momentum=None,
                 leapfrog_steps=50,
                 leapfrog_stepsize=.01):
        super(ClassifierHamiltonian, self).__init__()
        self.classifier = classifier
        self.leapfrog_steps = leapfrog_steps
        self.leapfrog_stepsize = leapfrog_stepsize
        self.momentum = momentum

    def _initialize(self, observations, **kwargs):
        # Initialize the momentum distribution.
        self._allocate_momentum(observations)

    def _allocate_momentum(self, observations):
        if self.momentum is None:
            dimensions = observations[0].dim()
            if dimensions == 0:
                momentum = Normal(0, 1)
            else:
                dimensions = len(observations[0])
                loc = torch.zeros(dimensions)
                scale = torch.eye(dimensions)
                momentum = MultivariateNormal(loc, scale)
            self.momentum = momentum

    def dU(self, observations, theta):
        return self.classifier.grad_log_likelihood(observations, theta)

    def K(self, momentum):
        return (.5 * momentum ** 2).sum()

    def dK(self, momentum):
        momentum = momentum.detach()
        momentum.requires_grad = True
        energy = self.K(momentum)
        energy.backward()
        gradient = momentum.grad.detach().clone()

        return gradient

    def step(self, observations, theta):
        theta = theta.squeeze().detach()
        momentum = self.momentum.sample()
        accepted = False

        m_next = momentum.clone().view(-1)
        t_next = theta.clone().view(-1)
        # Leapfrog integration
        eta = self.leapfrog_stepsize / 2
        for step in range(self.leapfrog_steps):
            m_next -= eta * self.dU(observations, t_next)
            t_next += self.leapfrog_stepsize * self.dK(m_next)
            m_next -= eta * self.dU(observations, t_next)
        K_initial = self.K(momentum)
        K_final = self.K(m_next)
        log_ratio = self.classifier.log_likelihood_ratio(observations, theta, t_next)
        rho = (log_ratio + K_initial - K_final).exp().item()
        probability = min([1, rho])
        u = np.random.uniform()
        if u <= probability:
            accepted = True
            theta = t_next

        return theta, probability, accepted
