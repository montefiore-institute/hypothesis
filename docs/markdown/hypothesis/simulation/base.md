Module hypothesis.simulation.base
=================================

Classes
-------

`BaseSimulator()`
:   Base simulator class.
    
    A simulator defines the implicit forward model.
    
    Example usage of a potential simulator implementation:
    
        simulator = MySimulator()
        inputs = prior.sample((10,)) # Draw 10 samples from the prior.
        outputs = simulator(inputs)
    
    In principle, this corresponds to sampling from the joint $$\vartheta,x\sim p(\vartheta)p(x\vert\vartheta)$$,
    where $$p(x\vert\vartheta)$$ is the likelihood-model implicitely defined through the simulator.
    
    .. note::
    
        The ``inputs`` and ``outputs`` variable name in most simulator denote
        their position with respect to the simulation model. ``inputs`` are
        typically free parameters of the simulation model which sample
        (or produce deterministically) ``outputs``.
    
    .. note::
    
        Although it is possibly to supply a batch of inputs, it should be
        noted that these are currently `not` parallelized.

    ### Descendants

    * hypothesis.benchmark.mg1.simulator.MG1BenchmarkSimulator
    * hypothesis.benchmark.spatialsir.simulator.SSIRBenchmarkSimulator
    * hypothesis.benchmark.tractable.simulator.TractableBenchmarkSimulator
    * hypothesis.benchmark.weinberg.simulator.WeinbergBenchmarkSimulator

    ### Methods

    `forward(self, **kwargs)`
    :   Defines the computation of the forward model at every call.
        
        .. note::
        
            Should be overridden by all subclasses.

    `terminate(self)`
    :   Terminates the simulator and cleans up possible contexts.
        
        .. note::
        
            Should be overridden by subclasses with a simulator state requiring graceful exits.