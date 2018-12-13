"""
Datasets API.
"""

import numpy as np
import os
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



class SimulatorDataset(HypothesisDataset):

    def __init__(self, simulator, prior, size=100000):
        super(SimulationDataset, self).__init__()
        self.prior = prior
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
        self.base = path
        self.base_x = path + "/x/"
        self.base_theta = path + "/theta/"
        self.num_files = len([x for x in os.listdir(self.base_theta) if os.path.isfile(self.base_theta + x)])
        self.buffer_block = 0
        self.buffer_theta, self.buffer_x = self.load_block(self.buffer_block)
        self.block_observations = len(self.buffer_theta)
        self.size = self.block_observations * self.num_files

    def load_block(self, index):
        # Load the thetas.
        index = str(index)
        with np.load(self.base_theta + index + ".npz") as theta_data:
            thetas = theta_data["arr_0"]
        # Load the observations.
        with np.load(self.base_x + index + ".npz") as x_data:
            xs = x_data["arr_0"]

        return thetas, xs

    def length(self):
        return self.size

    def sample(self):
        u = np.random.randint(0, self.size)
        return self.__getitem__(u)

    def __getitem__(self, index):
        # Check if the block is buffered.
        block_index = int(index / self.block_observations)
        if block_index != self.buffer_block:
            b_theta, b_x = self.load_block(block_index)
            self.buffer_theta = b_theta
            self.buffer_x = b_x
            self.buffer_block = block_index
        # Load the data from the buffer.
        data_index = index % self.block_observations
        theta = self.buffer_theta[data_index]
        x = self.buffer_x[data_index]

        return theta, x
