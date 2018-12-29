"""Common utility methods."""

import numpy as np
import torch



def sample(x, num_samples):
    with torch.no_grad():
        n = x.size(0)
        indices = torch.tensor(np.random.randint(0, n, num_samples))
        samples = x[indices]

    return samples
