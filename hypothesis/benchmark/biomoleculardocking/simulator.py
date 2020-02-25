import hypothesis
import numpy as np
import torch

from hypothesis.simulation import Simulator as BaseSimulator
from torch.distributions.beta import Beta
from torch.distributions.beta import Normal



class BiomolecularDockingSimulator(BaseSimulator):

    def __init__(self):
        super(BiomolecularDockingSimulator, self).__init__()
        self.r_bottom = Beta(4, 96)
        self.r_ee50 = Normal(-50, 15 ** 2)
        self.r_slope = Normal(-0.15, 0.1 ** 2)
        self.r_top = Beta(25, 75)

    def simulate(self, theta, psi):
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, inputs, experimental_configurations=None):
        outputs = []

        n = len(inputs)
        for index in range(n):
            theta = inputs[index]
            if experimental_configurations is not None:
                psi = experimental_configurations[index]
                x = self.simulate(theta, psi)
            else:
                x = self.simulate(theta, None)
            outputs.append(x.view(-1, 1))
        outputs = torch.cat(outputs, dim=0)

        return outputs
