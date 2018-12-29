"""Approximate Bayesian Computation"""

import numpy as np
import torch

from hypothesis.inference import Method



class ApproximateBayesianComputation(Method):
    r"""Vanilla Approximate Bayesian Computation

    Arguments:
        prior (Distribution): Prior distribution over the parameters of interest
        model (Simulator): TODO
        summary (lambda): Function to generate the summary statistic
        distance (lambda): Function expressing the distance between two given summary statistics
        epsilon (float): Acceptance threshold
    """

    KEY_NUM_SAMPLES = "samples"

    def __init__(self, prior, model, summary, distance, epsilon=.01):
        self.distance = distance
        self.epsilon = epsilon
        self.model = model
        self.prior = prior
        self.summary = summary
        self.summary_observations = None

    def sample(self, observations):
        num_observations = observations.size(0)
        sample = None
        while not sample:
            theta = self.prior.sample()
            inputs = theta.repeat(num_observations)
            outputs = self.model(theta)
            summary_outputs = self.summary(outputs)
            distance = self.distance(self.summary_observations, summary_outputs)
            if distance < self.epsilon:
                sample = theta

        return sample

    def infer(self, observations, **kwargs):
        self.summary_observations = self.summary(observations)
        num_samples = int(kwargs[self.KEY_NUM_SAMPLES])
        for sample_index in range(num_samples):
            sample = self.sample(observations)
            samples.append(sample)

        return samples
