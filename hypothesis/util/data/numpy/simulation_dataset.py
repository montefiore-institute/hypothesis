import numpy as np
import os
import torch

from torch.utils.data import Dataset
from hypothesis.util.data.numpy import Storage



class SimulationDataset(Dataset):
    r""""""

    def __init__(self, inputs, outputs):
        super(SimulationDataset, self).__init__()
        self.storage_inputs = Storage(inputs)
        self.storage_outputs = Storage(outputs)

    def __len__(self):
        return len(self.storage_inputs)

    def __del__(self):
        r""""""
        if hasttr(self, "storage_inputs") and self.storage_inputs is not None:
            self.storage_inputs.close()
            self.storage_outputs.close()

    def __getitem__(self, index):
        r""""""
        inputs = self.storage_inputs[index]
        outputs = self.storage_outputs[index]

        return inputs, outputs
