r"""Utilities for the Weinberg benchmark.

"""

import hypothesis as h
import torch


@torch.no_grad()
def Prior():
    r"""Returns a prior ``Uniform(0.25, 2.0)`` over the Fermi constant."""
    lower = torch.tensor(0.25).float()
    upper = torch.tensor(2.0).float()
    lower = lower.to(h.accelerator)
    upper = upper.to(h.accelerator)
    return torch.distributions.uniform.Uniform(lower, upper)


@torch.no_grad()
def PriorExperiment():
    r"""Returns a Prior ``Uniform(40.0, 50.0)`` over
    the experimental design space and represents the beam energy in GeV.

    """
    lower = torch.tensor(40.0).float()
    upper = torch.tensor(50.0).float()
    lower = lower.to(h.accelerator)
    upper = upper.to(h.accelerator)
    return torch.distributions.uniform.Uniform(lower, upper)  # KeV


@torch.no_grad()
def Truth():
    r"""Returns the true Fermi constant for this benchmark problem: ``1.0``."""
    return torch.tensor(1.0).float()
