import numpy as np
import os
import torch

from torch.utils.data import Dataset



class GeneratorDataset(Dataset):
    r""""""

    def __init__(self, model, prior, size=10000):
        super(ModelDataset, self).__init__()
        self.prior = prior
        self.model = model
        self.size = size

    def sample(self):
        x = self.prior.sample()
        y = self.model(x)

        return x, y

    def __getitem__(self, index):
        return self.sample()

    def __len__(self):
        return self.size



class SimulationDataset(Dataset):
    r""""""

    def __init__(self, path):
        super(SimulationDataset, self).__init__()
        self.base = path
        self.base_x = path + "/x/"
        self.base_theta = path + "/theta/"
