Module hypothesis.benchmark.spatialsir.util
===========================================
Utilities for the Weinberg benchmark.

Functions
---------

    
`Prior()`
:   Returns a uniform prior between 0 and 1 over the infection and
    recovery rate (encoded in this order).

    
`PriorExperiment()`
:   Returns a Prior ``Uniform(0.0, 10.0)`` over
    the experimental design space (measurement time).
    
    By default, the simulator will draw samples from
    this distribution to draw experimental configurations.

    
`Truth()`
:   Returns the true infection and recovery rate of this
    benchmark problem: ``(0.8, 0.2)``.