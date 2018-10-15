"""
Simulator base.

TODO Write docs.
"""

import torch



class Simulator(torch.nn.Module):

    def __init__(self):
        super(Simulator, self).__init__()

    def forward(self, thetas):
        raise NotImplementedError

    def terminate(self):
        raise NotImplementedError


class NoiseSimulator(torch.nn.Module):

    def __init__(self, simulator, sigma=1.):
        super(NoiseSimulator, self).__init__()
        self._simulator = simulator
        self._sigma = sigma

    def forward(self, thetas):
        thetas, x_thetas = self._simulator(thetas)
        x_thetas = x_thetas + self._sigma * torch.rand_like(x_thetas)

        return thetas, x_thetas
