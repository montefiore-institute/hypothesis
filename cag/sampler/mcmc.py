"""
Markov Chain Monte Carlo.
"""

import numpy as np
import torch

from cag.sampler import Sampler



class MetropolisHastings(Sampler):

    def __init__(self, simulator,
                 transition,
                 likelihood):
        super(MetropolisHastings, self).__init__(simulator)
        self.transition = transition
        self.likelihood = likelihood

    def infer(x_o, num_samples=1):
        raise NotImplementedError


class MetropolisHastingsApproximateLikelihoodRatios(Sampler):

    def __init__(self, simulator,
                 classifier,
                 transition):
        super(MetropolisHastingsApproximateLikelihoodRatios, self).__init__(simulator)
        self.classifier = classifier
        self.batch_size = batch_size
        self.transition = transition

    def infer(x_o, num_steps=1000):
        raise NotImplementedError
