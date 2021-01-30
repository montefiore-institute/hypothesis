import argparse
import hypothesis as h
import hypothesis.workflow as h
import os
import shutil

from hypothesis.workflow import SimulateTrainTestWorkflow
from hypothesis.benchmark.spatialsir import Prior
from hypothesis.benchmark.spatialsir import Simulator

parser = argparse.ArgumentParser()
parser.add_argument("--n", default=10000, type=int, help="Number of simulations to draw (default: 10000).")
parser.add_argument("--blocksize", default=1000, type=int, help="Size of a single block, i.e., number of simulations in a single block (default: 1000).")
arguments, _ = parser.parse_known_args()

prior = Prior
simulator = Simulator
workflow = SimulateTrainTestWorkflow(prior, simulator,
    directory="data",
    n_train=arguments.n,
    n_test=arguments.n,
    blocksize=arguments.blocksize)

# Build the workflow
# This will seed the workflow context in Hypothesis.
workflow.build()
