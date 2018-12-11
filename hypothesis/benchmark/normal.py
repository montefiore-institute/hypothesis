"""
Normal benchmarking problem.
"""

import torch

from hypothesis.simulation import Simulator



def simulator(thetas):
    with torch.no_grad():
        normal = torch.distributions.normal.Normal(thetas.view(-1), 1.)
        samples = normal.sample().view(-1, 1)

    return thetas, samples


def allocate_observations(theta, num_observations=100000):
    theta = torch.tensor(theta).float().view(-1)
    _, observations = simulator(torch.cat([theta] * num_observations))

    return theta, observations


class NormalSimulator(Simulator):

    def __init__(self):
        super(NormalSimulator, self).__init__()

    def forward(self, thetas):
        thetas = thetas.view(-1, 1)
        return simulator(thetas)

    def terminate(self):
        pass
