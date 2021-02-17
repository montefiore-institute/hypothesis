Module hypothesis.benchmark.spatialsir.simulator
================================================
Simulator definition of the SSIR benchmark.

Classes
-------

`SSIRBenchmarkSimulator(initial_infections_rate=3, shape=(100, 100), default_measurement_time=1.0, step_size=0.01)`
:   The simulation model generated a grid-world of susceptible,
    infected, and recovered individuals. This information is encoded in 3
    individual channels. Based on these parameters, and the infection and recovery rate,
    the model describes the evolution of an infection through this grid-like world.
    The disease spreads spatially, and is initialized with various number of
    initial infectious clusters, parameterized through a Poisson distribution.

    ### Ancestors (in MRO)

    * hypothesis.simulation.base.BaseSimulator

    ### Methods

    `simulate(self, theta, psi)`
    :