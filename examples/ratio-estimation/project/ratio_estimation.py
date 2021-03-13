import hypothesis as h
import numpy as np
import torch

from hypothesis.nn.ratio_estimation import RatioEstimatorEnsemble
from hypothesis.nn.ratio_estimation import build_mlp_estimator
from hypothesis.stat import highest_density_level
from hypothesis.util.data import NamedDataset
from torch.utils.data import TensorDataset


# Define the involved random variables and their shapes
random_variables = {
    "inputs": (1,),
    "outputs": (1,)}
denominator = "inputs|outputs"


BaseRatioEstimator = build_mlp_estimator(
    random_variables=random_variables,
    denominator=denominator)


class RatioEstimator(BaseRatioEstimator):

    def __init__(self):
        super(RatioEstimator, self).__init__(
            activation=torch.nn.SELU,
            trunk=[128, 128])


def Prior():
    return torch.distributions.uniform.Uniform(-10, 10)


prior = Prior()
extent = torch.linspace(-10, 10, 1000)
extent = extent.to(h.accelerator)


@torch.no_grad()
def compute_log_posterior(estimator, observable):
    observables = observable.repeat(1000, 1)
    log_posterior = log_pdf(estimator, extent, observables)

    return log_posterior


@torch.no_grad()
def log_pdf(estimator, inputs, outputs):
    inputs = inputs.view(-1, 1)
    outputs = outputs.view(-1, 1)
    assert inputs.shape[0] == outputs.shape[0]
    log_prior = prior.log_prob(inputs)
    log_mi = estimator.log_ratio(inputs=inputs, outputs=outputs)
    log_posterior = log_prior + log_mi

    return log_posterior


@torch.no_grad()
def compute_coverage(estimator, sample_joint, alpha=0.05):
    truth = sample_joint["inputs"]
    observable = sample_joint["outputs"]
    posterior = compute_log_posterior(estimator, observable).exp()
    hdl = highest_density_level(posterior, alpha=alpha)
    density_truth = log_pdf(estimator, truth, observable).exp().item()

    return int(density_truth >= hdl)


class Dataset(NamedDataset):

    def __init__(self, n):
        # Simulate
        prior = Prior()
        x = prior.sample((n,)).view(-1, 1).numpy()
        y = np.random.normal(size=n).reshape(-1, 1) + x
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
        super(DatasetTest, self).__init__(25000)
