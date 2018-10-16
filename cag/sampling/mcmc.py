"""
Markov Chain Monte Carlo.
"""

import numpy as np
import torch

from cag.sampling import Sampler



class MetropolisHastings(Sampler):

    def __init__(self, likelihood,
                 transition,
                 warmup_steps=10):
        super(MetropolisHastings, self).__init__()
        self.transition = transition
        self.likelihood = likelihood
        self._warmup_steps = int(warmup_steps)
        self._epsilon = 10e-7

    def _warmup(self, x, p_x):
        # Apply the burn-in / warm-up period.
        for step in range(self._warmup_steps):
            x, p_x = self.step(x, p_x)

        return x, p_x

    def step(self, x, p_x):
        accepted = False

        while not accepted:
            x_next = self.transition.sample(x)
            p_x_next = self.likelihood(x_next)
            u = np.random.uniform()
            p = (p_x_next / (p_x + self._epsilon))
            if not self.transition.is_symmetric():
                p *= (self.transition.log_prob(x_next, x) / (self.transition.log_prob(x, x_next) + self._epsilon))
            alpha = min([1, p])
            if u <= alpha:
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
        x, p_x = self._warmup(x, p_x)
        samples.append(x)
        # Start the sampling procedure.
        for step in range(num_samples - 1):
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
