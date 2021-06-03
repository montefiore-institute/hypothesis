r"""Utilities for the Poisson benchmark.

"""

import hypothesis as h

from torch.distributions.uniform import Uniform



def Prior():
    epsilon = 10e-5
    lower = torch.tensor(0).float()
    upper = torch.tensor(5).float()
    upper += epsilon
    lower = lower.to(h.accelerator)
    upper = upper.to(h.accelerator)

    return Uniform(lower, upper)
