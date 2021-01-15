"""Console script for hypothesis."""

import argparse
import hypothesis as h
import sys


def main():
    modules = ["config", "merge", "prune", "version", "workflow"]
    # Check if a command line option has been specified.
    if len(sys.argv) == 1 or not any(m in sys.argv for m in modules):
        show_help_and_exit()

    # Define the mapping between the command and executable functions.
    mapping = {
        "config": execute_config,
        "merge": execute_merge,
        "prune": execute_prune,
        "version": execute_version,
        "workflow": execute_workflow}
    # Execute the command, if it exists.
    mapping[sys.argv[1]]()


def execute_config():
    raise Exception("TODO")


def execute_merge():
    import hypothesis.bin.io.merge
    hypothesis.bin.io.merge.main()


def execute_prune():
    import hypothesis.bin.io.prune
    hypothesis.bin.io.prune.main()


def execute_version():
    print(h.__version__)


def execute_workflow():
    raise Exception("TODO")


def show_help_and_exit():
    help = r"""hypothesis """ + h.__version__ + """

Modules:
  config     Configuration module, adjust your hypothesis configuration.
  execute    Executes the specified workflow.
  merge      Utility program to merge data files.
  prune      Utility program to prune data files.
  version    Displays the current version of the hypothesis CLI.
"""
    print(help)
    sys.exit(0)


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
