Module hypothesis.benchmark.tractable.util
==========================================
Utilities for the tractable benchmark.

Functions
---------

    
`Prior()`
:   

    
`Truth()`
:   

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