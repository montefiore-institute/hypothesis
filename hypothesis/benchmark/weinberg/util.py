r"""Utilities for the Weinberg benchmark.

"""

import torch


@torch.no_grad()
def Prior():
    r"""Returns a prior over the Fermi constant."""
    return torch.distributions.uniform.Unifom(0.25, 2.0)


@torch.no_grad()
def PriorExperiment():
    r"""Prior over the experimental design space (the beam-energy)."""
    return torch.distributions.uniform.Uniform(40.0, 50.0)  # KeV


@torch.no_grad()
def Truth():
    return torch.tensor(1.0)
