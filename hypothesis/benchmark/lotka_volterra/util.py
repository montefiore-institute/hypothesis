r"""Utilities for the Lotka-Volterra population benchmark.

"""

import torch


@torch.no_grad()
def Prior():
    lower = torch.tensor(4 * [-10]).float()  # In ln-scale
    upper = torch.tensor(4 * [1]).float()    # In ln-scale

    return torch.distributions.uniform.Uniform(lower, upper)
