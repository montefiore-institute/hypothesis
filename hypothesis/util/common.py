"""Common utility methods."""

import numpy as np
import torch



def sample(x, num_samples):
    with torch.no_grad():
        n = x.size(0)
        indices = torch.tensor(np.random.randint(0, n, num_samples))
        samples = x[indices]

    return samples


def parse_argument(**kwargs, key, default, type):
    # Check if the key has been specified.
    if key in kwargs.keys():
        argument = type(kwargs[key])
    else:
        argument = default

    return argument
