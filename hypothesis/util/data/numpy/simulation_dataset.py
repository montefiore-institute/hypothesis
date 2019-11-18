import numpy as np
import os
import torch

from torch.utils.data import Dataset
from hypothesis.util.data.numpy import InMemoryStorage
from hypothesis.util.data.numpy import PersistentStorage



class SimulationDataset(Dataset):
    r""""""

    def __init__(self, inputs, outputs, in_memory=False):
        super(SimulationDataset, self).__init__()
        if in_memory:
            self.storage_inputs = InMemoryStorage(inputs)
            self.storage_outputs = InMemoryStorage(outputs)
        else:
            self.storage_inputs = PersistentStorage(inputs)
            self.storage_outputs = PersistentStorage(outputs)

    def __len__(self):
        return len(self.storage_inputs)

    def __del__(self):
        r""""""
        if hasattr(self, "storage_inputs") and self.storage_inputs is not None:
            self.storage_inputs.close()
            self.storage_outputs.close()

    def __getitem__(self, index):
        r""""""
        inputs = self.storage_inputs[index]
        outputs = self.storage_outputs[index]

        return inputs, outputs
