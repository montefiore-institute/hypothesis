import hypothesis
import numpy as np
import torch

from hypothesis.nn.neuromodulation import BaseNeuromodulatedModule



class NeuromodulatedReLU(BaseNeuromodulatedModule):

    def __init__(self, controller, inplace=False):
        super(NeuromodulatedReLU, self).__init__(
            controller=controller,
            activation=torch.nn.ReLU,
            **{"inplace": inplace})
