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
import tempfile


def main():
    prepare_directory()
    executor = load_default_executor()
    arguments = parse_arguments()
    if arguments.slurm:
        executor = "slurm"
    if arguments.local:
        executor = "local"
    executors = {
        "slurm": execute_slurm,
        "local": execute_local}
    # Check what module needs to be executed
    ## Execution
    if arguments.execute is not None:
        script = arguments.execute[1]
        exec(open(script).read(), globals())
        executors[executor](arguments)
    elif arguments.list is not None:
        list_store(arguments)


def store_directory():
    return os.path.expanduser("~") + "/.hypothesis/workflow"


def list_store(arguments):
    paths = os.listdir(store_directory())
    for p in paths:
        if os.path.isdir(p):
            print(p)


def prepare_directory():
    path = store_directory()
    if not os.path.exists(path):
        os.makedirs(path)


def execute_slurm(arguments):
    logging.info("Using the Slurm workflow backend.")
    store = tempfile.mkdtemp(dir=store_directory())
    hypothesis.workflow.slurm.execute(
        directory=arguments.directory,
        environment=arguments.environment,
        store=store,
        cleanup=not arguments.no_cleanup)


def execute_local(arguments):
    logging.info("Using the local workflow backend.")
    hypothesis.workflow.local.execute()


def load_default_executor():
    if hypothesis.workflow.slurm.slurm_detected():
        return "slurm"
    else:
        return "local"


def parse_arguments():
    parser = argparse.ArgumentParser()
    # Slurm backend configuration
    parser.add_argument("--directory", type=str, default=".", help="Directory to generate the Slurm submission scripts (default: '.').")
    parser.add_argument("--environment", type=str, default="base", help="Anaconda environment to execute the Slurm tasks with (default: base).")
    parser.add_argument("--no-cleanup", action="store_true", help="Disables the cleanup subroutine of the Slurm submission scripts.")
    # Logging options
    parser.add_argument("--level", default="info", type=str, help="Minimum logging level (default: warning) (options: debug, info, warning, error, critical).")
    # Executor backend
    parser.add_argument("--slurm", action="store_true", help="Force the usage the Slurm executor backend.")
    parser.add_argument("--local", action="store_true", help="Force the usage the local executor backend.")
    # Workflow modules
    parser.add_argument("cancel", nargs='?', help="Cancel the specified workflow (only for the Slurm backend).")
    parser.add_argument("clean", nargs='?', help="Clean up old workflows from the store (only for the Slurm backend).")
    parser.add_argument("delete", nargs='?', help="Delete and cancel old or existing workflows (only for the Slurm backend).")
    parser.add_argument("execute", nargs='?', help="Executes the specified workflow.")
    parser.add_argument("list", nargs='?', help="Lists all workflows in the store (only for the Slurm backend).")
    parser.add_argument("status", nargs='?', help="Utility to monitor the status of workflows (only for the Slurm backend).")
    # Parse the arguments
    arguments = parser.parse_intermixed_args()
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
