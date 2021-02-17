Module hypothesis.benchmark.mg1.util
====================================
Utilities for the M/G/1 benchmark.

Functions
---------

    
`Prior()`
:   Returns a uniform prior between ``(0, 0, 0)`` and
    `(10, 10, 1/3)`.

    
`Truth()`
:   Returns the true queuing model parameters: ``(1, 5, 0.2)``.

Classes
-------

`Uniform(lower, upper)`
:   Generates uniformly distributed random samples from the half-open interval
    ``[low, high)``.
    
    Example::
    
        >>> m = Uniform(torch.tensor([0.0]), torch.tensor([5.0]))
        >>> m.sample()  # uniformly distributed in the range [0.0, 5.0)
        tensor([ 2.3418])
    
    Args:
        low (float or Tensor): lower range (inclusive).
        high (float or Tensor): upper range (exclusive).

    ### Ancestors (in MRO)

    * torch.distributions.uniform.Uniform
    * torch.distributions.distribution.Distribution

    ### Methods

    `log_prob(self, sample)`
    :   Returns the log of the probability density/mass function evaluated at
        `value`.
        
        Args:
            value (Tensor):