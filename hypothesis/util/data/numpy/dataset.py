import numpy as np
import os
import torch

from hypothesis.util.data.numpy import InMemoryStorage
from hypothesis.util.data.numpy import PersistentStorage
from torch.utils.data import Dataset as BaseDataset



class Dataset(BaseDataset):
    r""""""

    def __init__(self, *paths, in_memory=False):
        super(Dataset, self).__init__()
        if in_memory:
            storage_type = InMemoryStorage
        else:
            storage_type = PersistentStorage
        self.storages = [storage_type(path) for path in paths]

    def __getitem__(self, index):
        return tuple(torch.from_numpy(storage[index]).unsqueeze(0) for storage in self.storages)

    def __del__(self):
        for index in range(len(self.storages)):
            storage = self.storages[index]
            storage.close()
            del storage
            self.storages[index] = None
        self.storages = None

    def __len__(self):
        return len(self.storages[0])
