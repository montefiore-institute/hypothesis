import hypothesis
import numpy as np
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
    def _retrieve(self, index):
        elements = []
        for storage_index in range(self.num_storages):
            elements.append(self.storages[storage_index][index].unsqueeze(dim=0))

        return tuple(elements)

    @torch.no_grad()
    def put(self, transition):
        for index in range(self.num_storages):
            self.storages[index].append(transition[index].squeeze())

    def size(self):
        return len(self.storages[0])

    @torch.no_grad()
    def sample(self, batch_size=hypothesis.default.batch_size):
        indices = np.random.randint(0, self.size())
        tensors = [[] for _ in range(self.num_storages)]
        experiences = [self._retrieve(index) for index in indices]
        for experience in experiences:
            for storage_index, element in enumerate(experience):
                tensors[storage_index].append(element)
        for index in range(self.num_storages):
            tensors[index] = torch.cat(tensors[index], dim=0)

        return tuple(tensors)

    def __getitem__(self, index):
        return self._retrieve(index)

    def __len__(self):
        return self.size()
