"""
Markov Chain Monte Carlo methods for inference.
"""

import numpy as np
import torch


from cag.inference import Method



class MetropolisHastings(Method):

    def __init__(self, simulator,
                 likelihood,
                 transition,
                 summary,
                 warmup_steps=10):
        super(MetropolisHastings, self).__init__(simulator)
        self.transition = transition
        self.likelihood = likelihood
        self.summary = summary
        self._warmup_steps = int(warmup_steps)

    def _warmup(self, x, p_x):
        raise NotImplementedError

    def step(self, x, p_x):
        raise NotImplementedError

    def infer(self, x_o, initializer, num_samples):
        raise NotImplementedError


class LikelihoodFreeMetropolisHastings(Method):

    def __init__(self, simulator,
                 classifier,
                 transition,
                 warmup_steps=10,
                 simulations=10000):
        super(LikelihoodFreeMetropolisHastings, self).__init__(simulator)
        self.classifier = classifier
        self.transition = transition
        self._warmup_steps = warmup_steps
        self._simulations = simulations
        self._epsilon = 10e-7
        self._o_discriminator = None

    def _simulate(self, theta):
        theta = torch.cat([theta] * self._simulations, dim=0)
        _, x_theta = self.simulator(theta)

        return x_theta

    def _warmup(self, x_o, theta, x_theta):
        for step in range(self._warmup_steps):
            theta, x_theta = self.step(x_o, theta, x_theta)

        return theta, x_theta

    def step(self, theta, x_theta):
        accepted = False

        while not accpeted:
            # Sample the next theta from the proposal.
            theta_next = self.transition.sample(theta)
            x_theta_next = self._simulate(theta_next)
            likelihood_ratio = self._likelihood_ratio(x_o, theta_next, x_theta_next, theta, x_theta)
            if not self.transition.is_symmetric():
                t_theta_next = self.transition.log_prob(theta_next, theta)
                t_theta = self.transition.log_prob(theta, theta_next)
                p *= (t_theta_next / (t_theta + self._epsilon))
            alpha = min([1, p])
            u = np.random.uniform()
            if u <= alpha:
                theta = theta_next
                accpeted = True

        return theta, x_theta

    def infer(self, x_o, initializer, num_samples):
        samples = []

        # Draw a random initial sample from the initializer.
        theta = initializer.sample().detach().view(-1)
        x_theta = self._simulate(theta)
        # MCMC burn-in.
        theta, x_theta = self._warmup(x_o, theta, x_theta)
        samples.append(theta)
        # Start the sampling procedure.
        for step in range(num_samples - 1):
            theta, x_theta = self.step(x_o, theta, x_theta)
            samples.append(theta)

        return torch.cat(samples, dim=0)
