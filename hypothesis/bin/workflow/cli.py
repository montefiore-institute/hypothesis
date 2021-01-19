r"""A utility program to handle Hypothesis workflows.

"""

import argparse
import glob
import hypothesis as h
import hypothesis.workflow as w
import hypothesis.workflow.local
import hypothesis.workflow.slurm
import logging
import os
import sys
import tempfile

from stat import S_ISREG, ST_CTIME, ST_MODE


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
    script = arguments.args[1]
    exec(open(script).read(), globals(), locals())
    executors[executor](arguments)


def prepare_directory():
    path = store_directory()
    if not os.path.exists(path):
        os.makedirs(path)


def assert_slurm_detected():
    if not hypothesis.workflow.slurm.slurm_detected():
        logging.critical("Slurm is not configured on this system!")
        sys.exit(0)


def execute_slurm(arguments):
    assert_slurm_detected()
    logging.info("Using the Slurm workflow backend.")
    if arguments.name is None:
        store = tempfile.mkdtemp(dir=store_directory())
    else:
        store = store_directory() + '/' + arguments.name
        if os.path.exists(store):
            logging.error("The workflow name with `" + arguments.name + "` already exists.")
            sys.exit(0)
        else:
            os.makedirs(store)
    hypothesis.workflow.slurm.execute(
        directory=arguments.directory,
        environment=arguments.environment,
        store=store,
        cleanup=arguments.cleanup)


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
    parser.add_argument("--description", type=str, default=None, help="Provide a description to the workflow (default: none).")
    parser.add_argument("--directory", type=str, default=".", help="Directory to generate the Slurm submission scripts (default: '.').")
    parser.add_argument("--environment", type=str, default=None, help="Anaconda environment to execute the Slurm tasks with (default: none).")
    parser.add_argument("--name", type=str, default=None, help="Determines the name of the workflow (default: random).")
    parser.add_argument("--cleanup", action="store_true", help="Enables the cleanup subroutine of the Slurm submission scripts.")
    # Logging options
    parser.add_argument("--level", default="info", type=str, help="Minimum logging level (default: warning) (options: debug, info, warning, error, critical).")
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

    return arguments


if __name__ == "__main__":
    main()
