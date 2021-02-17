"""Stochastic Spatial Susceptible Infected Recovered benchmark.

This problem setting is concerned with the computation of a posterior
over the infection and recovery rate \(\\vartheta\), conditioned on an observable \(x\),
representing a grid-world of susceptible, infected, and recovered individuals.
This information is encoded in 3 individual channels. Based on these parameters,
the model describes the evolution of an infection through this grid-like world.
The disease spreads spatially, and is initialized with various number of
initial infectious clusters, parameterized through a Poisson distribution.
"""

from .simulator import SSIRBenchmarkSimulator
from .util import Prior
from .util import PriorExperiment
from .util import Truth
