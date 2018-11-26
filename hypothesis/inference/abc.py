"""
Approximate Bayesian Computation.
"""

import numpy as np
import torch

from hypothesis.inference import Method



class ApproximateBayesianComputation(Method):

    KEY_NUM_SAMPLES = "samples"

    def __init__(self, prior, model, summary, distance, epsilon=.01):
        super(ApproximateBayesianComputation, self).__init__()
        self.prior = prior
        self.model = model
        self.summary = summary
        self.distance = distance
        self._epsilon = .01

    def epsilon(self):
        return self._epsilon

    def sample(self, observations):
        sample = None

        s_o = self.summary(observations)
        num_observations = observations.size(0)
        while not sample:
            theta = self.prior.sample()
            x_theta = self.model(theta, num_observations)
            s_x = self.summary(x_theta)
            d = self.distance(s_o, s_x)
            if d < self._epsilon:
                sample = theta

        return sample

    def procedure(self, observations, **kwargs):
        samples = []

        num_samples = int(kwargs[self.KEY_NUM_SAMPLES])
        for sample_index in range(num_samples):
            sample = self.sample(observations)
            samples.append(sample)

        return samples
