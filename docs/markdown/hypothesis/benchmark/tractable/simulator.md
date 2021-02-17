Module hypothesis.benchmark.tractable.simulator
===============================================
Hallo wereld?

Classes
-------

`Simulator()`
:   Base simulator class.
    
    A simulator defines the forward model.
    
    Example usage of a potential simulator implementation:
    
    .. code-block:: python
    
        simulator = MySimulator()
        inputs = prior.sample((10,)) # Draw 10 samples from the prior.
        outputs = simulator(inputs)
    
    .. note::
    
        The ``inputs`` and ``outputs`` variable name in most simulator denote
        their position with respect to the simulation model. ``inputs`` are
        typically free parameters of the simulation model which sample
        (or produce deterministically) ``outputs``.
    
    .. note::
    
        Although it is possibly to supply a batch of inputs, it should be
        noted that these are currently `not` parallelized.
    
    Simulation model associated with the tractable benchmark.

    ### Ancestors (in MRO)

    * hypothesis.simulation.base.BaseSimulator
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :