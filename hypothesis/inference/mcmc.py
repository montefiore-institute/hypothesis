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



class MarkovChainMonteCarlo(Method):

    KEY_INITIAL_THETA = "theta_0"
    KEY_SAMPLES = "samples"
    KEY_BURNIN_SAMPLES = "burnin_samples"

    def __init__(self):
        super(MarkovChainMonteCarlo, self).__init__()

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
            hypothesis.call_hooks(hypothesis.hooks.post_step, self, theta=theta, probability=probability, accepted=accepted)
            samples.append(theta)
            probabilities.append(probability)
            acceptances.append(accepted)

        return samples, probabilities, acceptances

    def infer(self, observations, **kwargs):
        burnin_samples = None
        burnin_probabilities = None
        burnin_acceptances = None

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
        samples, probabilities, acceptances = self.sample(observations, theta_0, num_samples)
        # Allocate the chain summary.
        chain = Chain(samples, probabilities, acceptances,
                      burnin_samples, burnin_probabilities, burnin_acceptances)

        return chain



class RatioMetropolisHastings(MarkovChainMonteCarlo):

    def __init__(self, ratio, transition):
        super(RatioMetropolisHastings, self).__init__()
        self.likelihood_ratio = ratio
        self.transition = transition

    def step(self, observations, theta):
        accepted = False

        with torch.no_grad():
            theta_next = self.transition.sample(theta).squeeze()
            lr = self.likelihood_ratio(observations, theta, theta_next)
            if not self.transition.is_symmetric():
                t_next = self.transition.log_prob(theta, theta_next).exp()
                t_current = self.transition.log_prob(theta_next, theta).exp()
                p = (t_next / (t_current + epsilon))
            else:
                p = 1
            probability = min([1, lr * p])
            u = np.random.uniform()
            if u <= probability:
                accepted = True
                theta = theta_next

        return theta, probability, accepted



class MetropolisHastings(RatioMetropolisHastings):

    def __init__(self, log_likelihood, transition):
        # Define the ratio function in terms of the log-likelihood.
        def likelihood_ratio(self, observations, theta, theta_next):
            likelihood_current = log_likelihood(observations, theta)
            likelihood_next = log_likelihood(observations, theta_next)
            lr = likelihood_next - likelihood_current

            return lr.exp()
        # Initialize the parent with the ratio-method.
        super(MetropolisHastings, self).__init__(likelihood_ratio, transition)
