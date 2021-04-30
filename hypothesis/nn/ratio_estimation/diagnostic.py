r"""Diagnostics specific to ratio estimators.

"""

import hypothesis as h
import numpy as np


@torch.no_grad()
def underestimate_mutual_information(dataset_joint, r, n=10000):
    expectation = expectation_marginals_ratio(dataset_joint=dataset_joint, r=r, n=10000)

    return expectation <= 1.0


@torch.no_grad()
def overestimate_mutual_information(dataset_joint, r, n=10000):
    return not underestimate_mutual_information(dataset_joint=dataset_joint, r=r, n=n)


@torch.no_grad()
def expectation_marginals_ratio(dataset_joint, r, n=10000):
    required_random_variables = set(["inputs", "outputs"])
    assert required_random_variables.issubset(set(r.random_variables.keys()))
    index_range = np.arange(len(dataset_joint))
    indices = np.random.choice(index_range, n, replace=False)
    inputs = d[indices]["inputs"]
    inputs = inputs.to(h.accelerator)
    indices = np.random.choice(index_range, n, replace=False)
    outputs = d[indices]["outputs"]
    outputs = outputs.to(h.accelerator)
    ratio = r.log_ratio(inputs=inputs, outputs=outputs).exp()

    return ratio.mean()
