r"""Predefined workflows and utilities to build dependency graphs on your local
workstation and HPC clusters.

Workflows are acyclic computational graphs which essentially define a dependency graph
of procedures. These graph serve as an abstraction for `executors`, which execute the
workflow locally or generate the required code to run the computational graph on an
HPC cluster without having to write the necessarry supporting code. Everything can
directly be done in Python. These workflows also tie directly with the `hypothesis workflow`
CLI tool.
"""

context = None
executor = None

from .base import BaseWorkflow
from .decorator import *
from .util import *
from .workflow import *

import hypothesis.workflow.local
import hypothesis.workflow.slurm
