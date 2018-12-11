"""
Base modules for simulations.
"""

import torch



class Simulator(torch.nn.Module):

    def __init__(self):
        super(Simulator, self).__init__()

    def forward(self, thetas):
        """
        Method should return thetas, x_thetas.
        """
        raise NotImplementedError

    def terminate(self):
        raise NotImplementedError
