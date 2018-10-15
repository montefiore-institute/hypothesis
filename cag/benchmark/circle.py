"""
Deterministic circle problem.

The goal of this problem is the infer the x, y and radius of a single circle.
"""

import numpy as np
import torch

from cag.simulation import Simulator



def allocate_observations(theta, num_observations=100000):
    simulator = CircleSimulator()
    theta = torch.tensor(theta).float().view(1, 3)
    theta = torch.cat([theta] * num_observations)
    _, x_o = simulator(theta)

    return theta, x_o


class CircleSimulator(Simulator):

    def __init__(self, axial_resolution=64, epsilon=.025):
        super(CircleSimulator, self).__init__()
        x = np.linspace(-1, 1, axial_resolution)
        y = np.linspace(-1, 1, axial_resolution)
        X, Y = np.meshgrid(x, y)
        self._X = torch.from_numpy(X).float()
        self._Y = torch.from_numpy(Y).float()
        self._axial_resolution = axial_resolution
        self._epsilon = epsilon

    def _generate(self, r, x, y):
        M = (self._X - x) ** 2 + (self._Y + y) ** 2 - (r ** 2) < self._epsilon
        M = M.float().view(1, self._axial_resolution, self._axial_resolution)

        return M

    def forward(self, thetas):
        samples = []

        with torch.no_grad():
            batch_size = thetas.size(0)
            radius, position = thetas.split([1, 2], dim=1)
            X, Y = position.split(1, dim=1)
            for batch_index in range(batch_size):
                r = radius[batch_index]
                x = X[batch_index]
                y = Y[batch_index]
                samples.append(self._generate(r, x, y))
            samples = torch.cat(samples, dim=0).contiguous()

        return thetas, samples
