r"""Generate the physics of a hypothetical 2-D spherical world, and then generate
seismic events and detections.

Based on the codebase of: Nimar Arora https://github.com/nimar/seismic-2d/blob/master/generate.py
"""

import hypothesis
import numpy as np
import torch

from hypothesis.simulation import Simulator as BaseSimulator



class SeismicSimulator(BaseSimulator):

    def __init__(self):
        super(SeismicSimulator, self).__init__()

    def forward(self, inputs, experimental_configurations=None):
        raise NotImplementedError
