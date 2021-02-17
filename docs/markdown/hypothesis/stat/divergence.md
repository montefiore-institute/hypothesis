Module hypothesis.stat.divergence
=================================
Utilities to compute divergences between densities, or
samples of those densities.

Functions
---------

    
`jsd(samples_p: torch.Tensor, samples_q: torch.Tensor, model=None) ‑> torch.Tensor`
:   Computes the Jensen-Shannon Divergence between samples from \(p\) and \(q\).