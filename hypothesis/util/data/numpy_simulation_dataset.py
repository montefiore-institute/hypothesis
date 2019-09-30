import os
import torch

from torch.utils.data import Dataset



class NumpySimulationDataset(Dataset):
    r""""""

    def __init__(self, inputs, outputs):
        super(NumpySimulationDataset, self).__init__()
        if not os.path.exists(inputs) or not os.path.exists(outputs):
            raise Valuerror("Please specify a proper inputs or outputs path.")
        self.path_inputs = inputs
        self.path_outputs = outputs
        raise NotImplementedError
