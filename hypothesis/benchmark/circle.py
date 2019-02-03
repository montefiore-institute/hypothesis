"""
Circle problem.

The goal of this problem is to infer the x, y and radius of a single observed circle
as the simulations are deterministic.
"""

import numpy as np
import torch

from hypothesis.simulation import Simulator



def allocate_observations(theta):
    inputs = torch.tensor(theta).view(1, 3)
    simulator = CircleSimulator()
    output = simulator(inputs)

    return output



class CircleSimulator(Simulator):

    def __init__(self, resolution=32, epsilon=.025):
        super(CircleSimulator, self).__init__()
        x = np.linspace(-1, 1, resolution)
        y = np.linspace(-1, 1, resolution)
        X, Y = np.meshgrid(x, y)
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
        self.resolution = resolution
        self.epsilon = epsilon

    def generate(self, x, y, r):
        M = (self.X - x) ** 2 + (self.Y - y) ** 2 - (r ** 2) < self.epsilon
        M = M.float().view(-1, self.resolution, self.resolution)

        return M

    def forward(self, inputs):
        samples = []

        with torch.no_grad():
            batch_size = inputs.size(0)
            position, radius = inputs.split([2, 1], dim=1)
            X, Y = position.split(1, dim=1)
            for batch_index in range(batch_size):
                x = X[batch_index]
                y = Y[batch_index]
                r = radius[batch_index]
                samples.append(self.generate(x, y, r))
            samples = torch.cat(samples, dim=0).contiguous()

        return samples
