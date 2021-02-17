Module hypothesis.simulation.base
=================================

Classes
-------

`BaseSimulator()`
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

    * torch.nn.modules.module.Module

    ### Descendants

    * hypothesis.benchmark.mg1.simulator.Simulator
    * hypothesis.benchmark.spatialsir.simulator.Simulator
    * hypothesis.benchmark.tractable.simulator.Simulator
    * hypothesis.benchmark.weinberg.simulator.Simulator

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :

    ### Methods

    `forward(self, **kwargs) ‑> Callable[..., Any]`
    :   Defines the computation of the forward model at every call.
        
        .. note::
        
            Should be overridden by all subclasses.

    `terminate(self)`
    :   Terminates the simulator and cleans up possible contexts.
        
        .. note::
        
            Should be overridden by subclasses with a simulator state requiring graceful exits.