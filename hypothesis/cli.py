"""Main CLI interface for ``hypothesis``.

"""

import argparse
import hypothesis as h
import sys


def main():
    modules = ["merge", "prune", "version", "workflow"]
    # Check if a command line option has been specified.
    if len(sys.argv) == 1 or not any(m in sys.argv for m in modules):
        show_help_and_exit()

    # Define the mapping between the command and executable functions.
    mapping = {
        "merge": execute_merge,
        "prune": execute_prune,
        "version": execute_version,
        "workflow": execute_workflow}
    module = sys.argv[1]
    del sys.argv[:1]
    # Execute the command, if it exists.
    mapping[module]()


def execute_merge():
    import hypothesis.bin.io.merge
    hypothesis.bin.io.merge.main()


def execute_prune():
    import hypothesis.bin.io.prune
    hypothesis.bin.io.prune.main()


def execute_version():
    print(h.__version__)


def execute_workflow():
    import hypothesis.bin.workflow.cli
    hypothesis.bin.workflow.cli.main()


def show_help_and_exit():
    help = r"""hypothesis """ + h.__version__ + """

Modules:
  execute    Executes the specified workflow.
  merge      Utility program to merge data files.
  prune      Utility program to prune data files.
  version    Displays the current version of the hypothesis CLI.
  workflow   Utility software to aid with the execution of Hypothesis workflows.
"""
    print(help)
    sys.exit(0)


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
