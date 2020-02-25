import hypothesis
import numpy as np
import torch

from hypothesis.simulation import Simulator as BaseSimulator



class BiomolecularDockingSimulator(BaseSimulator):

    MIN_PSI = -75.0
    MAX_PSI = 0.0
    EXPERIMENTAL_SPACE = 100

    def __init__(self, default_experimental_design=torch.zeros(EXPERIMENTAL_SPACE)):
        super(BiomolecularDockingSimulator, self).__init__()
        self.default_experimental_design = default_experimental_design

    def simulate(self, theta, psi):
        bottom = theta[0].item()
        ee50 = theta[1].item()
        slope = theta[2].item()
        top = theta[3].item()
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
                x = self.simulate(theta, self.default_experimental_design)
            outputs.append(x.view(-1, 1))
        outputs = torch.cat(outputs, dim=0)

        return outputs
