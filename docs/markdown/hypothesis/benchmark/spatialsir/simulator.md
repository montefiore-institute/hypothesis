Module hypothesis.benchmark.spatialsir.simulator
================================================
This problem setting is concerned with the computation of a posterior
over the infection and recovery rate $\vartheta$, conditioned on an observable $x$,
representing a grid-world of susceptible, infected, and recovered individuals.
This information is encoded in 3 individual channels. Based on these parameters,
the model describes the evolution of an infection through this grid-like world.
The disease spreads spatially, and is initialized with various number of
initial infectious clusters, parameterized through a Poisson distribution.

Classes
-------

`Simulator(initial_infections_rate=3, shape=(100, 100), default_measurement_time=1.0, step_size=0.01)`
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

    ### Methods

    `simulate(self, theta, psi)`
    :