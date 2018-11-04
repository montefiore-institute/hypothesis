"""
Utility methods for `hypothesis`.
"""

import numpy as np
import torch



def sample_distribution(distribution, num_samples):
    size = torch.Size([num_samples])
    samples = distribution.sample(size)

    return samples


def sample(x, num_samples):
    with torch.no_grad():
        indices = torch.tensor(np.random.randint(0, x.size(0), num_samples))
        samples = x[indices]

    return samples
