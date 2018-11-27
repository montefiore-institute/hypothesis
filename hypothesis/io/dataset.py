"""
Datasets API.
"""

import numpy as np
import h5py as h5
import torch

from torch.utils.data import Dataset



class HypothesisDataset(Dataset):

    def __init__(self):
        super(HypothesisDataset, self).__init__()

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return self.size()

    def size(self):
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

    def size(self):
        return self.data.size(0)



class H5Dataset(HypothesisDataset):

    def __init__(self, path, name_input, name_output):
        super(H5Dataset, self).__init__()
        self.file = h5.File(self.path, 'r')
        self.x = self.file.get(name_input)
        self.y = self.file.get(name_output)
        self.size = self.file.len()

    def __getitem__(self, index):
        x = torch.from_numpy(self.x[index]).float()
        y = torch.from_numpy(self.y[index]).float()

        return (x, y)

    def size(self):
        return self.size



class SimulationDataset(HypothesisDataset):

    def __init__(self, distribution, simulator, size):
        super(SimulationDataset, self).__init__()
        self.distribution = distribution
        self.simulator = simulator
        self.size = size

    def size(self):
        return self.size

    def sample(self):
        with torch.no_grad():
            theta = self.distribution().sample()
            theta, x_theta = self.simulator(theta)

        return theta, x_theta

    def __getitem__(self, index):
        return self.sample()



class ReferenceDataset(HypothesisDataset):

    def __init__(self, reference, simulator, size):
        super(ReferenceDataset, self).__init__()
        self.reference = reference
        self.simulator = simulator
        self.size = size

    def size(self):
        return self.size

    def sample(self):
        with torch.no_grad():
            theta, x_theta = self.simulator(reference)

        return theta, x_theta
