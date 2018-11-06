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
        self.log_likelihood = log_likelihood
        self.transition = transition

    def step(self, observations, theta):
        raise NotImplementedError

    def procedure(self, observations, **kwargs):
        # Initialize the sampling procedure.
        theta_0 = kwargs[self.KEY_INITIAL_THETA]
        num_samples = int(kwargs[self.KEY_NUM_SAMPLES])
        if self.KEY_BURN_IN_STEPS in kwargs.keys():
            burnin_steps = int(kwargs[self.KEY_BURN_IN_STEPS])
        else:
            burnin_steps = 0
