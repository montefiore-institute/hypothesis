import torch

from hypothesis.simulation import Simulator



def simulator(inputs):
    with torch.no_grad():
        inputs = inputs.view(-1)
        poisson = torch.distributions.poisson.Poisson(inputs)
        outputs = poisson.sample().view(-1, 1)

    return outputs


def allocate_observations(theta, observations=10000):
    inputs = torch.tensor(theta).float().view(-1).repeat(observations)
    outputs = simulator(inputs)

    return outputs


class PoissonSimulator(Simulator):

    def __init__(self):
        super(PoissonSimulator, self).__init__()

    def forward(self, inputs):
        return simulator(inputs)
