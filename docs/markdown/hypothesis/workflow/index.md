Module hypothesis.workflow
==========================
Predefined workflows and utilities to build dependency graphs on your local
workstation and HPC clusters.

Workflows are acyclic computational graphs which essentially define a dependency graph
of procedures. These graph serve as an abstraction for `executors`, which execute the
workflow locally or generate the required code to run the computational graph on an
HPC cluster without having to write the necessarry supporting code. Everything can
directly be done in Python. These workflows also tie directly with the `hypothesis workflow`
CLI tool.

Sub-modules
-----------
* hypothesis.workflow.base
* hypothesis.workflow.decorator
* hypothesis.workflow.graph
* hypothesis.workflow.local
* hypothesis.workflow.slurm
* hypothesis.workflow.util
* hypothesis.workflow.workflow