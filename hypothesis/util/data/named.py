import torch

from hypothesis.util import is_tensor
from torch.utils.data import Dataset as BaseDataset


class BaseNamedDataset(BaseDataset):

    def __init__(self, keys):
        self._keys = keys

    def keys(self):
        return self._keys

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class NamedDataset(BaseNamedDataset):

    def __init__(self, **datasets):
        super(NamedDataset, self).__init__(list(datasets.keys()))
        self._datasets = datasets
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


class NamedSubDataset(BaseNamedDataset):

    def __init__(self, source_dataset, source_indices):
        assert instance(source_dataset, BaseNamedDataset)
        super(NamedSubDataset, self).__init__(source_dataset.keys())
        self._source_dataset = source_dataset
        self._source_indices = source_indices

    def __getitem__(self, index):
        return self._source_dataset[self._source_indices[index]]

    def __len__(self):
        return len(self._indices)
