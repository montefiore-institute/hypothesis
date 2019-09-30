import torch

from hypothesis.simulation import Simulator
from torch.utils.data import Dataset



class SimulatorDataset(Dataset):
    r"""

    Todo:
        Write docs.
    """

    def __init__(self, simulator, prior, size=1000000):
        super(SimulatorDataset, self).__init__()
        self.prior = prior
        self.simulator = simulator
        self.size = int(size)

    def __getitem(self, index):
        r"""
        Todo:
            Write docs.
        """
        inputs = prior.sample(torch.Size([1]))
        outputs = self.simulator(inputs)

        return inputs, outputs

    def __len__(self):
        r"""
        Todo:
            Write docs.
        """
        return self.size
