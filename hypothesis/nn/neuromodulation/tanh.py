import hypothesis
import numpy as np
import torch

from hypothesis.nn.neuromodulation import BaseNeuromodulatedModule



class NeuromodulatedTanh(BaseNeuromodulatedModule):

    def __init__(self, controller):
        super(NeuromodulatedTanh, self).__init__(
            controller=controller,
            activation=torch.nn.Tanh)
