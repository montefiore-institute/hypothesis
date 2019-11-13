import numpy as np
import os
import torch

from hypothesis.util.data.numpy import Storage
from torch.utils.data import Dataset as BaseDataset



class Dataset(BaseDataset):
    r""""""

    def __init__(self, *paths):
        super(NumpyDataset, self).__init__()
        self.storages = [Storage(path) for path in paths]

    def __getitem__(self, index):
        return tuple(storage[index] for storage in self.storages)

    def __del__(self):
        for index in len(self.storages):
            storage = self.storages[index]
            storage.close()
            del storage
            self.storages[index] = None
        self.storages = None

    def __len__(self):
        return len(self.storages[0])
