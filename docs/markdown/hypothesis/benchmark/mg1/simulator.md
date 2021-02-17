Module hypothesis.benchmark.mg1.simulator
=========================================
Simulation model of the M/G/1 queuing model.

This model describes a queuing system of continuously arriving jobs by a
single server. The time it takes to process every job is uniformly
distributed in the interval :math:`[\theta_1, \theta_2]`. The arrival
between two consecutive jobs is exponentially distributed according to
the rate :math:`\theta_3`. That is, for
every job :math:`i` we have the processing time :math:`p_i` , an arrival
time :math:`a_i` and the time :math:`l_i` at which the job left the queue.

Classes
-------

`Simulator(percentiles=5, steps=50)`
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
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * hypothesis.simulation.base.BaseSimulator
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :