r"""A utility program to handle Hypothesis workflows.

"""

import argparse
import hypothesis as h
import hypothesis.workflow as w
import hypothesis.workflow.local
import hypothesis.workflow.slurm
import os
import sys


def main():
    executor = load_default_executor()
    arguments = parse_arguments()
    if arguments.local:
        executor = w.local.execute
    if arguments.slurm:
        executor = w.slurm.execute


def load_default_executor():
    if hypothesis.workflow.slurm.slurm_detected():
        executor = w.slurm.execute
    else:
        executor = w.local.execute

    return executor


def parse_arguments():
    parser = argparse.ArgumentParser()
    # Slurm backend configuration
    # Executor backend
    parser.add_argument("--slurm", action="store_true", help="Force the usage the Slurm executor backend.")
    parser.add_argument("--local", action="store_true", help="Force the usage the local executor backend.")
    arguments, _ = parser.parse_known_args()

    return arguments


if __name__ == "__main__":
    main()
