Module hypothesis.benchmark.weinberg.simulator
==============================================
Simulator definition of the Weinberg benchmark.

Classes
-------

`WeinbergBenchmarkSimulator(default_beam_energy=40.0, num_samples=1)`
:   This is a simulation of high energy particle collisions $$e^+e^- \to \mu^+ \mu^-.$$
    The angular distributions of the particles can be used to measure the Weinberg angle
    in the standard model of particle physics. If you get a PhD in particle physics,
    you may learn how to calculate these distributions and interpret those equations to
    learn that an effective way to infer this parameter is to run your particle accelerator
    with a beam energy just above or below half the $Z$ boson mass (i.e. the optimal $\phi$
    is just above and below 45 GeV).
    
    Adapted from https://github.com/cranmer/active_sciencing/blob/master/demo_weinberg.ipynb
    
    Original implementation by Lucas Heinrich and Kyle Cranmer.
    
    ```python
    from hypothesis.benchmark.weinberg import Prior
    from hypothesis.benchmark.weinberg import Simulator
    
    prior = Prior()
    simulator = Simulator()
    
    inputs = prior.sample((10,))  # Draw 10 samples from the prior
    outputs = simulator(inputs)
    
    # You can also batch with respect to the experimental configurations
    from hypothesis.benchmark.weinberg import PriorExperiment
    
    prior_experiment = PriorExperiment()
    beam_energies = prior_experiment.sample((10,))
    outputs = simulator(inputs, beam_energies)
    ```

    ### Ancestors (in MRO)

    * hypothesis.simulation.base.BaseSimulator

    ### Class variables

    `GFNom`
    :

    `MZ`
    :

    ### Methods

    `forward(self, inputs, experimental_configurations=None, **kwargs)`
    :   Executes the forward pass of the simulation model.
        
        :param inputs: Free parameters (the Fermi constant).
        :param experimental_configurations: Optional experimental
                                            parameters describing the beam energy.
        
        .. note::
        
            This method accepts a batch of corresponding inputs and optional
            experimental configuration pairs.