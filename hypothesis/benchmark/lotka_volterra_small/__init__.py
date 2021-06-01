r"""Benchmark problem based on the Predator-Prey population evolution model.

This particular problem setting only focusses on the Predator parameters,
while effictively marginalizing over the Prey paramters.
"""

from .simulator import LotkaVolterraBenchmarkSimulator as Simulator
from .util import Prior
