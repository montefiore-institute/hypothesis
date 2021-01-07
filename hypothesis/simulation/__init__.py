r""":mod:`hypothesis.simulation` is a submodule containing the
base simulator architecture and utilities to execute efficient simulations.
Every forward model requires to be wrapped in a class which inherits from
:class:`hypothesis.simulation.BaseSimulator`.

"""

from .base import BaseSimulator
