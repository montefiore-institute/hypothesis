import hypothesis
import numpy as np
import random
import torch

from torch.utils.data import Dataset



class ExperienceReplayBuffer(Dataset):

    def __init__(self):
        self.storage = []

    def put(self, transition):
        self.storage.append(transition)

    def sample(self, batch_size=hypothesis.default.batch_size):
        experiences = random.sample(self.storage, k=batch_size)

        return experiences

    def size(self):
        return len(self.storage)

    def __getitem__(self, index):
        return self.storage[index]

    def __len__(self):
        return self.size()
