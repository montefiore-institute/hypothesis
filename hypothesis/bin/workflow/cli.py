r"""A utility program to handle Hypothesis workflows.

"""

import argparse
import coloredlogs
import glob
import hypothesis as h
import hypothesis.workflow as w
import hypothesis.workflow.local
import hypothesis.workflow.slurm
import imp
import logging
import os
import shutil
import sys
import tempfile


def main():
    prepare_directory()
    arguments = parse_arguments()
    # Module mapping
    mapping = {
        "delete": delete_workflow,
        "clean": clean_workflows,
        "execute": execute_workflow,
        "status": workflow_status,
        "list": list_store}
    # Check what module needs to be executed
    if len(arguments.args) > 0:
        command = arguments.args[0]
        if command in mapping:
            f = mapping[command]
            f(arguments)


def store_directory():
    return os.path.expanduser("~") + "/.hypothesis/workflow"


def list_store(arguments):
    store = store_directory()
    paths = os.listdir(store_directory())
    entries = []
    for p in paths:
        if os.path.isdir(store + '/' + p):
            print(p)


def delete_workflow(arguments):
    assert_slurm_detected()
    query = store_directory() + '/' + arguments.args[1] + '*' + "/slurm_jobs"
    paths = glob.glob(query)
    if len(paths) == 0:
        logging.critical("The specified workflow could not be found. Try `list`.")
        sys.exit(0)
    workflow_path = paths[0]
    workflow_directory = os.path.dirname(os.path.realpath(workflow_path))
    f = open(workflow_path, 'r')
    lines = f.readlines()
    f.close()
    logging.info("Cancelling Slurm jobs related to workflow.")
    for identifier in lines:
        identifier = identifier[:-1]
        if h.util.is_integer(identifier):
            os.system("scancel " + identifier)
            logging.info("Cancelled Slurm job " + identifier + ".")
    shutil.rmtree(workflow_directory)


def clean_workflows(arguments):
    raise NotImplementedError


def workflow_status(arguments):
    assert_slurm_detected()
    raise NotImplementedError


def execute_workflow(arguments):
    executor = load_default_executor()
    if arguments.slurm:
        executor = "slurm"
    if arguments.local:
        executor = "local"
    executors = {
        "slurm": execute_slurm,
        "local": execute_local}
    imp.load_source('__main__', arguments.args[1])
    executors[executor](arguments)


def prepare_directory():
    path = store_directory()
    if not os.path.exists(path):
        os.makedirs(path)


def assert_slurm_detected():
    if not hypothesis.workflow.slurm.slurm_detected():
        logging.critical("Slurm is not configured on this system!")
        sys.exit(1)


def execute_slurm(arguments):
    assert_slurm_detected()
    if not arguments.parsable:
        logging.info("Using Slurm backend.")
    if arguments.name is None:
        store = tempfile.mkdtemp(dir=store_directory())
        name = os.path.basename(store)
        if arguments.parsable:
            logging.info(name)
        else:
            logging.info("Executing workflow " + name)
    else:
        store = store_directory() + '/' + arguments.name
        if os.path.exists(store):
            logging.critical("The workflow name with `" + arguments.name + "` already exists.")
            sys.exit(1)
        else:
            os.makedirs(store)
    if arguments.directory is None:
        directory = os.path.basename(store)
    else:
        directory = arguments.directory
    hypothesis.workflow.slurm.execute(
        directory=directory,
        environment=arguments.environment,
        partition=arguments.partition,
        store=store,
        cleanup=arguments.cleanup)


def execute_local(arguments):
    if not arguments.parsable:
        logging.info("Using local backend.")
    hypothesis.workflow.local.execute()


def load_default_executor():
    if hypothesis.workflow.slurm.slurm_detected():
        return "slurm"
    else:
        return "local"


def parse_arguments():
    parser = argparse.ArgumentParser()
    # Slurm backend configuration
    parser.add_argument("--cleanup", action="store_true", help="Enables the cleanup subroutine of the Slurm submission scripts.")
    parser.add_argument("--description", type=str, default=None, help="Provide a description to the workflow (default: none).")
    parser.add_argument("--directory", type=str, default='.', help="Directory to generate the Slurm submission scripts (default: none).")
    parser.add_argument("--environment", type=str, default=None, help="Anaconda environment to execute the Slurm tasks with (default: none).")
    parser.add_argument("--name", type=str, default=None, help="Determines the name of the workflow (default: random).")
    parser.add_argument("--partition", type=str, default=None, help="Slurm partition to deploy the job to (default: none).")
    # Logging options
    parser.add_argument("--parsable", action="store_true", help="Outputs to stdout should be easily parsable (default: false).")
    parser.add_argument("--format", type=str, default="%(message)s", help="Format of the logger.")
    parser.add_argument("--level", default="info", type=str, help="Minimum logging level (default: warning) (options: debug, info, warning, error, critical).")
    parser.add_argument("-v", action="store_true", help="Enable verbosity in the details of the log messages (default: false).")
    # Executor backend
    parser.add_argument("--slurm", action="store_true", help="Force the usage the Slurm executor backend.")
    parser.add_argument("--local", action="store_true", help="Force the usage the local executor backend.")
    # Workflow modules
    parser.add_argument("args", nargs='*', help="Slurm backend utilities, currently supporting: delete, clean, execute, list.")
    # Parse the arguments
    arguments, _ = parser.parse_known_args()
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
    if arguments.v:
        arguments.format = "%(asctime)-15s " + arguments.format
    coloredlogs.install(fmt=arguments.format)

    return arguments


if __name__ == "__main__":
    main()
