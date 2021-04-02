r"""Tractable benchmark

This smaller version of `hypothesis.benchmark.tractable` marginalizes
over the 5th-parameter. The total dimensionality of the problem size
is therefore reduces to 4.
"""

from .simulator import TractableBenchmarkSimulator as Simulator
from .util import Prior
from .util import Truth
