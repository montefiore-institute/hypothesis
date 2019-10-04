r"""Approximate Bayesian Computation"""

import numpy as np
import torch



class ApproximateBayesianComputation:

    def __init__(self, prior, summary):
        super(ApproximateBayesianComputation, self).__init__()
        self.prior = prior
        self.summary = summary

    def sample(self, observation, num_samples):
        raise NotImplementedError
