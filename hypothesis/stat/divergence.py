r"""Utilities to compute divergences between densities, or
samples of those densities.

"""

import torch



def jsd(samples_p: torch.Tensor, samples_q: torch.Tensor, model=None) -> torch.Tensor:
    r"""Computes the Jensen-Shannon Divergence between samples from \(p\) and \(q\).

    """
    raise NotImplementedError
