import torch

from hypothesis.util import is_tensor
from torch.utils.data import Dataset as BaseDataset


class NamedDataset(BaseDataset):

    def __init__(self, **datasets):
        super(NamedDataset, self).__init__()
        self._datasets = datasets
        self._keys = list(datasets.keys())
        self._reference = self._datasets[self._keys[0]]

    def __getitem__(self, index):
        sample = {}
        for k in self._keys:
            element = self._datasets[k][index]
            if is_tensor(element):
                sample[k] = element
            else:
                sample[k] = element[0] # 0-indexing for tuple extraction.

        return sample

    def __len__(self):
        return len(self._reference)
