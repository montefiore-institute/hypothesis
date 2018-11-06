"""
Markov Chain Monte Carlo methods for inference.
"""

import copy
import numpy as np
import torch

from hypothesis.inference import Method
from hypothesis.inference import SimulatorMethod



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
        theta_next = self.transition.sample(theta)
        likelihood_current = self.log_likelihood(theta, observations)
        likelihood_next = self.log_likelihood(theta_next, observations)
        lr = likelihood_next - likelihood_current
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
            theta_0, acceptance = self.step(observations, theta_0)
            thetas.append(theta_0.squeeze())
            probabilities.append(acceptance)

        return thetas, probabilities

    def procedure(self, observations, **kwargs):
        # Initialize the sampling procedure.
        theta_0 = kwargs[self.KEY_INITIAL_THETA]
        num_samples = int(kwargs[self.KEY_NUM_SAMPLES])
        if self.KEY_BURN_IN_STEPS in kwargs.keys():
            burnin_steps = int(kwargs[self.KEY_BURN_IN_STEPS])
        else:
            burnin_steps = 0
        # Start the burnin procedure.
        thetas, probabilities = self.run_chain(theta_0, observations, burnin_steps)
        # Start sampling form the MH chain.
        thetas, probabilities = self.run_chain(theta_0, observations, num_samples)

        return torch.tensor(thetas), probabilities
