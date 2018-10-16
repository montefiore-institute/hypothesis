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


class ClassifierMetropolisHastings(Method):

    def __init__(self, simulator,
                 classifier,
                 transition,
                 warmup_steps=10,
                 simulations=10000):
        super(ClassifierMetropolisHastings, self).__init__(simulator)
        self.classifier = classifier
        self.transition = transition
        self._warmup_steps = warmup_steps
        self._simulations = simulations

    def infer(self, x_o, initializer, num_samples):
        raise NotImplementedError
