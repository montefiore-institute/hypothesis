import hypothesis
import numpy as np
import torch

from hypothesis.nn.neuromodulation import BaseNeuromodulatedModule



class NeuromodulatedELU(BaseNeuromodulatedModule):

    def __init__(self, controller, inplace=False):
        super(NeuromodulatedELU, self).__init__(
            controller=controller,
            activation=torch.nn.ELU,
            **{"inplace": inplace})
