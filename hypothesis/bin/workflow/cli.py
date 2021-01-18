r"""A utility program to handle Hypothesis workflows.

"""

import argparse
import hypothesis as h
import hypothesis.workflow as w
import hypothesis.workflow.local
import hypothesis.workflow.slurm
import logging
import os
import sys


def main():
    executor = load_default_executor()
    arguments = parse_arguments()
    if arguments.slurm:
        executor = "slurm"
    if arguments.local:
        executor = "local"
    executors = {
        "slurm": execute_slurm,
        "local": execute_local}
    if len(arguments.args) > 0:
        script = arguments.args[0]
        exec(open(script).read())
        executors[executor](arguments)


def execute_slurm(arguments):
    logging.info("Using the Slurm workflow backend.")


def execute_local(arguments):
    logging.info("Using the local workflow backend.")


def load_default_executor():
    if hypothesis.workflow.slurm.slurm_detected():
        return "slurm"
    else:
        return "local"


def parse_arguments():
    parser = argparse.ArgumentParser()
    # Slurm backend configuration
    parser.add_argument("--directory", type=str, default=None, help="Directory to generate the Slurm submission scripts.")
    # Logging options
    parser.add_argument("--level", default="warning", type=str, help="Minimum logging level (default: warning) (options: debug, info, warning, error, critical).")
    # Executor backend
    parser.add_argument("--slurm", action="store_true", help="Force the usage the Slurm executor backend.")
    parser.add_argument("--local", action="store_true", help="Force the usage the local executor backend.")
    # Remaining position arguments
    parser.add_argument("args", nargs=argparse.REMAINDER)
    # Parse the arguments
    arguments = parser.parse_args()
    arguments.level = arguments.level.lower()
    logging_level_mapping = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL}
    if arguments.level in logging_level_mapping.keys():
        level = logging_level_mapping[arguments.level]
        logging.getLogger().setLevel(level)

    return arguments


if __name__ == "__main__":
    main()
