"""
Simulator base.

TODO Write docs.
"""

import torch



class Simulator(torch.nn.Module):

    def __init__(self):
        super(Simulator, self).__init__()

    def forward(self, thetas):
        raise NotImplementedError

    def terminate(self):
        raise NotImplementedError
