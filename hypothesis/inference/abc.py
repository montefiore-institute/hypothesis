"""
Approximate Bayesian Computation.
"""

import numpy as np
import torch

from hypothesis.inference import Method



class ApproximateBayesianComputation(Method):


    def __init__(self):
        super(ApproximateBayesianComputation, self).__init__()

    def procedure(self, observations, **kwargs):
        raise NotImplementedError
