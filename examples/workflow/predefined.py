import argparse
import hypothesis as h
import hypothesis.workflow as h
import os
import shutil

from hypothesis.workflow import SimulationWorkflow
from hypothesis.benchmark.spatialsir import Prior
from hypothesis.benchmark.spatialsir import Simulator

parser = argparse.ArgumentParser()
parser.add_argument("--n", default=10000, type=int, help="Number of simulations to draw (default: 10000).")
parser.add_argument("--blocksize", default=1000, type=int, help="Size of a single block, i.e., number of simulations in a single block (default: 1000).")
parser.add_argument("--clean", action="store_true", help="Clean the simulated data files (default: false).")
arguments, _ = parser.parse_known_args()

if arguments.clean:
    if os.path.exists("blocks"):
        shutil.rmtree("blocks")
    if os.path.exists("inputs.npy"):
        os.remove("inputs.npy")
    if os.path.exists("outputs.npy"):
        os.remove("outputs.npy")

prior = Prior
simulator = Simulator
workflow = SimulationWorkflow(prior, simulator, n=arguments.n, blocksize=arguments.blocksize)

# Build the workflow
workflow.build()
