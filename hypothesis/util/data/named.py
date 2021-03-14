import torch

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
            sample[k] = self._datasets[k][index]

        return sample

    def __len__(self):
        return len(self._reference)
