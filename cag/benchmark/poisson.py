"""
Poisson benchmarking problem.
"""

import torch

from cag.simulation import Simulator



def simulator(thetas):
    with torch.no_grad():
        poisson = torch.distributions.poisson.Poisson(thetas.view(-1))
        samples = poisson.sample().view(-1, 1)

    return thetas, samples


def allocate_observations(theta, num_observations=100000):
    theta = torch.tensor(theta).view(-1)
    _, x_o = simulator(torch.cat([theta] * num_observations))

    return theta, x_o


class PoissonSimulator(Simulator):

    def __init__(self):
        super(PoissonSimulator, self).__init__()

    def forward(self, thetas):
        return simulator(thetas)

    def terminate(self):
        pass
