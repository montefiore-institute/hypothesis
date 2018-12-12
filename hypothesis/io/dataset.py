"""
Datasets API.
"""

import numpy as np
import torch

from torch.utils.data import Dataset



class HypothesisDataset(Dataset):

    def __init__(self):
        super(HypothesisDataset, self).__init__()

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return self.length()

    def length(self):
        raise NotImplementedError

    def sample(self):
        u = np.random.randint(0, self.size())

        return self.__getitem__(u)



class TensorDataset(HypothesisDataset):

    def __init__(self, data):
        super(TensorDataset, self).__init__()
        self.data = data

    def __getitem__(self, index):
        return self.data[index].detach()

    def length(self):
        return self.data.size(0)



class GenerativeDataset(HypothesisDataset):

    def __init__(self, distribution, simulator, size=100000):
        super(SimulationDataset, self).__init__()
        self.distribution = distribution
        self.simulator = simulator
        self.size = size

    def length(self):
        return self.size

    def sample(self):
        with torch.no_grad():
            theta = self.distribution.sample()
            theta, x_theta = self.simulator(theta)

        return theta, x_theta

    def __getitem__(self, index):
        return self.sample()



class SimulationDataset(HypothesisDataset):

    def __init__(self, path):
        super(SimulationDataset, self).__init__()
        raise NotImplementedError

    def length(self):
        return self.size

    def sample(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError
