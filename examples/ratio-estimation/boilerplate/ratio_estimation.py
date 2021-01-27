import hypothesis as h
import numpy as np
import torch

from hypothesis.nn.ratio_estimation import build_mlp_estimator
from hypothesis.util.data import NamedDataset
from torch.utils.data import TensorDataset


# Define the involved random variables and their shapes
random_variables = {
    "inputs": (1,),
    "outputs": (1,)}
denominator = "inputs|outputs"


RatioEstimator = build_mlp_estimator(
    random_variables=random_variables,
    denominator=denominator)


class Dataset(NamedDataset):

    def __init__(self, n):
        # Simulate
        x = np.random.uniform(-15, 15, n)
        y = np.random.random(n) + x
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        # Create datasets
        inputs = TensorDataset(torch.from_numpy(x).float())
        outputs = TensorDataset(torch.from_numpy(y).float())
        super(Dataset, self).__init__(
            inputs=inputs,
            outputs=outputs)


class DatasetTrain(Dataset):

    def __init__(self):
        super(DatasetTrain, self).__init__(100000)


class DatasetTest(Dataset):

    def __init__(self):
        super(DatasetTest, self).__init__(100000)
