import hypothesis
import numpy as np
import random
import torch

from torch.utils.data import Dataset



class ExperienceReplayBuffer(Dataset):

    def __init__(self, num_stores=4):
        # Check if a valid number of stores has been specified
        if num_stores <= 0:
            raise ValueError("A valid number of data storages (> 0) needs to be specified.")
        # Replay buffer state
        self.num_storages = num_stores
        self.storages = [[] for _ in range(self.num_storages)]

    @torch.no_grad()
    def _retriever(self, index):
        elements = []
        for storage_index in range(self.num_storages):
            elements.append(self.storages[storage_index][index].unsqueeze(dim=0))

        return tuple(elements)

    @torch.no_grad()
    def put(self, transition):
        for index in range(len(transition)):
            self.storages[index].append(transition[index].squeeze())

    def size(self):
        return len(self.storages[0])

    def __getitem__(self, index):
        return self._retriever(index)

    def __len__(self):
        return self.size()
