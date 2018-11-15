"""
Parallel methods for MCMC.
"""

import torch
import numpy as np
import torch.multiprocessing

from hypothesis.inference import Method as InferenceMethod


class InferenceEnsemble(InferenceMethod):

    def __init__(self, sampler, chains=2):
        super(Ensemble, self).__init__()
        self._sampler = sampler
        self._chains = chains

    def procedure(observations, **kwargs):
        raise NotImplementedError
