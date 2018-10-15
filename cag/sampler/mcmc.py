"""
Markov Chain Monte Carlo.
"""

import numpy as np
import torch

from cag.sampler import Sampler



class MetropolisHastings(Sampler):

    def __init__(self, simulator,
                 likelihood,
                 transition=None,
                 warmup_steps=10):
        super(MetropolisHastings, self).__init__(simulator)
        self.transition = transition
        self.likelihood = likelihood
        self._warmup = warmup_steps

    def _warmup(self, theta):
        # Apply the burn-in / warm-up period.
        for step in range(self._warmup):
            theta = self.step(theta)

    def step(self, x, p_x):
        accepted = False

        while not accepted:
            x_next = self.transition.sample(x)
            p_x_next = self.likelihood(x_next)
            u = np.random.uniform()
            if u <= p_x_next:
                x = x_next
                p_x = p_x_next
                accepted = True

        return x, p_x

    def sample(self, initializer, num_samples):
        samples = []

        # Draw a random initial sample from the initializer.
        x = initializer.sample().detach()
        p_x = self.likelihood(x)
        # Start the warmup period.
        x, p_x = self._warmup(x)
        samples.append(x)
        # Start the sampling procedure.
        for step in range(num_samples):
            x, p_x = self.step(x, p_x)
            samples.append(x)

        return torch.cat(samples, dim=0)


class ClassifierMetropolisHastings(Sampler):

    def __init__(self, simulator,
                 classifier,
                 transition):
        super(MetropolisHastingsApproximateLikelihoodRatios, self).__init__(simulator)
        self.classifier = classifier
        self.batch_size = batch_size
        self.transition = transition

    def sample(self):
        raise NotImplementedError
