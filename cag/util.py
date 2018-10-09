"""
Utility methods for `cag`.
"""

import numpy as np
import torch



def sample_distribution(distribution, num_samples):
    with torch.no_grad():
        size = torch.Size([num_samples])
        samples = distribution.sample(size)

    return samples



def sample(x, num_samples):
    with torch.no_grad():
        permutations = torch.randperm(x.size(0))
        indices = permutations[:num_samples]
        samples = x[indices]

    return samples
