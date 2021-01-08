r"""Utilities for the Weinberg benchmark.

"""

import torch


@torch.no_grad()
def Prior():
    r"""Returns a prior ``Uniform(0.25, 2.0)`` over the Fermi constant."""
    return torch.distributions.uniform.Uniform(0.25, 2.0)


@torch.no_grad()
def PriorExperiment():
    r"""Returns a Prior ``Uniform(40.0, 50.0)`` over
    the experimental design space and represents the beam energy in GeV.

    """
    return torch.distributions.uniform.Uniform(40.0, 50.0)  # KeV


@torch.no_grad()
def Truth():
    r"""Returns the true Fermi constant for this benchmark problem: ``1.0``."""
    return torch.tensor(1.0)
