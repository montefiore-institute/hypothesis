r"""``hypothesis.simulation`` is a package consisting of the base simulator
architecture and utilities to perform efficient simulations. Every forward
model needs to be wrapped in a class which inherits from
``hypothesis.simulation.Simulator``.
"""

from .base import Simulator
