import torch

from hypothesis.simulation import Simulator
from torch.distributions.normal import Normal



class NormalSimulator(Simulator):
    r"""

    Todo:
        Write method docs.
    """

    def __init__(self, uncertainty=1):
        super(NormalSimulator, self).__init__()
        self.uncertainty = float(uncertainty)

    def forward(self, inputs, designs=None):
        if designs is None:
            designs = self.uncertainty

        return Normal(inputs, designs).sample()
