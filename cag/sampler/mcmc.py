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

    def sample(self):
        raise NotImplementedError


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
