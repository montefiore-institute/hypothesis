import hypothesis
import numpy as np
import torch

from hypothesis.nn.neuromodulation import BaseNeuromodulatedModule



class NeuromodulatedSELU(BaseNeuromodulatedModule):

    def __init__(self, controller, inplace=False):
        super(NeuromodulatedSELU, self).__init__(
            controller=controller,
            activation=torch.nn.SELU,
            **{"inplace": inplace})
