r"""Utilities for the Lotka-Volterra population benchmark.

"""

import torch


@torch.no_grad()
def Prior():
    epsilon = 0.00001
    lower = torch.tensor(4 * [-4]).float()  # In ln-scale
    upper = torch.tensor(4 * [1]).float()  # In ln-scale
    upper += epsilon  # Account for half-open interval

    return torch.distributions.uniform.Uniform(lower, upper)
