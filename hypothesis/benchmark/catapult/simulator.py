import hypothesis
import numpy as np
import torch

from hypothesis.simulation import Simulator as BaseSimulator



class CatapultSimulator(BaseSimulator):

    def __init__(self, step_size=0.01):
        super(CatapultSimulator, self).__init__()
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, inputs, experimental_configurations=None):
        raise NotImplementedError
