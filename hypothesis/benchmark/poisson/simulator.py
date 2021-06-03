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
        outputs = []

        for rate in inputs:
            if rate <= 0:
                observable = torch.tensor(0).view(1, 1)
            else:
                poisson = torch.distributions.poisson.Poisson(rate.view(-1))
                observable = poisson.sample().view(1, 1)
            outputs.append(observable)

        return torch.cat(outputs, dim=0).float()
