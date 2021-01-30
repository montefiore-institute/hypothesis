r"""Hallo wereld?"""

import torch

from hypothesis.simulation import BaseSimulator
from torch.distributions.multivariate_normal import MultivariateNormal as Normal


class Simulator(BaseSimulator):

    def __init__(self):
        r"""Simulation model associated with the tractable benchmark.

        """
        super(Simulator, self).__init__()

    @torch.no_grad()
    def _generate(self, input):
        mean = torch.tensor([input[0], input[1]])
        scale = 1.0
        s_1 = input[2] ** 2
        s_2 = input[3] ** 2
        rho = input[4].tanh()
        covariance = torch.tensor([
            [scale * s_1 ** 2, scale * rho * s_1 * s_2],
            [scale * rho * s_1 * s_2, scale * s_2 ** 2]])
        normal = Normal(mean, covariance)
        x_out = normal.sample(torch.Size([4])).view(1, -1)

        return x_out

    @torch.no_grad()
    def forward(self, inputs, **kwargs):
        samples = []

        inputs = inputs.view(-1, 5)
        for input in inputs:
            x_out = self._generate(input)
            samples.append(x_out.view(1, -1))

        return torch.cat(samples, dim=0)
