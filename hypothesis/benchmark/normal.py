import torch

from hypothesis.simulation import Simulator



def simulator(inputs):
    with torch.no_grad():
        inputs = inputs.view(-1)
        normal = torch.distributions.normal.Normal(inputs, 1)
        outputs = normal.sample().view(-1, 1)

    return outputs


def allocate_observations(theta, observations=10000):
    inputs = torch.tensor(theta).float().view(-1).repeat(observations)
    outputs = simulator(inputs)

    return outputs


class NormalSimulator(Simulator):

    def __init__(self):
        super(NormalSimulator, self).__init__()

    def forward(self, inputs):
        return simulator(inputs)
