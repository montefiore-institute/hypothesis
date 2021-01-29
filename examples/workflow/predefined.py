import argparse
import hypothesis as h
import hypothesis.workflow as h

from hypothesis.workflow import SimulationWorkflow
from hypothesis.benchmark.spatialsir import Prior
from hypothesis.benchmark.spatialsir import Simulator

prior = Prior
simulator = Simulator
workflow = SimulationWorkflow(prior, simulator, n=10000)

# Build the workflow
workflow.build()
