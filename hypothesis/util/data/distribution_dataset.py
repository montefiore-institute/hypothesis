import torch

from torch.utils.data import Dataset



class DistributionDataset(Dataset):
    r""""""

    def __init__(self, distribution, size=1000000):
        self.distribution = distribution
        self.size = size

    def __getitem__(self, index):
        return self.distribution.sample()

    def __len__(self):
        return self.size
