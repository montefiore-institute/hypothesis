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
        if len(paths) > 1:
            self.storages = [storage_type(path) for path in paths]
            self.retriever = self._retrieve_multi_storage
        else:
            self.storage = storage_type(paths[0])
            self.retriever = self._retrieve_single_storage
            self.storages = [self.storage]

    def _retrieve_multi_storage(self, index):
        return tuple(storage[index].unsqueeze(0) for storage in self.storages)

    def _retrieve_single_storage(self, index):
        return self.storage[index]

    def __getitem__(self, index):
        return self.retriever(index)

    def __del__(self):
        if hasattr(self, "storages"):
            for index in range(len(self.storages)):
                storage = self.storages[index]
                storage.close()
                del storage
                self.storages[index] = None
            self.storages = None

    def __len__(self):
        return len(self.storages[0])
