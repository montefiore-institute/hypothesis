import torch

from torch.utils.data import Dataset



class SimulationTensorDataset(Dataset):
    r""""""

    def __init__(self, inputs, outputs):
        super(SimulationTensorDataset, self).__init__()
        self.inputs = inputs
        self.outputs = outputs

    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]

    def __len__(self):
        return self.inputs.shape[0]
