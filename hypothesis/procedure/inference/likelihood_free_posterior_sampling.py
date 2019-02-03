"""
Procedures for performing likelihood-free posterior sampling.
"""

import torch
import hypothesis

from hypothesis.inference import Method
from hypothesis.nn import ParameterizedClassifier
from hypothesis.train import ParameterizedClassifierTrainer
from torch.distributions.uniform import Uniform



class IterativeLikelihoodFreePosteriorSampling(Method):
    r"""Iterative likelihood-free posterior sampling"""

    KEY_CLASSIFIER = "classifier"
    KEY_PRIOR = "prior"
    KEY_TRAINER = "trainer"

    def __init__(self, rounds=2, sampler="hmc", offset=.1):
        super(IterativeLikelihoodFreePosteriorSampling, self).__init__()
        self.offset = offset
        self.rounds = rounds
        self.sampler = sampler

    def infer(self, observations, **kwargs):
        raise NotImplementedError
