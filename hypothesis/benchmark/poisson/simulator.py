r"""Simulator definition of the Poisson benchmark.

"""

import torch

from hypothesis.simulation import BaseSimulator



class PoissonBenchmarkSimulator(BaseSimulator):
    r"""Simulation model of the Poisson process.

    """

    def __init__(self):
        super(PoissonBenchmarkSimulator, self).__init__()

    @torch.no_grad()
    def forward(self, inputs):
         poisson = torch.distributions.poisson.Poisson(inputs.view(-1))

         return poisson.sample().view(-1, 1)
