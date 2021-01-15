"""Console script for hypothesis."""

import argparse
import hypothesis as h
import sys


def main():
    # Define the command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("merge", action="store_true", help="Utility to merge data files.")
    parser.add_argument("prune", action="store_true", help="Utility to prune data files.")
    parser.add_argument("version", action="store_true", help="Shows the version and commit of the hypothesis binary.")
    arguments, _ = parser.parse_known_args()

    # Check if a command line option has been specified.
    if len(sys.argv) == 1:
        show_help_and_exit(parser)

    # Define the mapping between the command and executable functions.
    mapping = {
        "merge": execute_merge,
        "prune": execute_prune,
        "version": execute_version}
    # Execute the command, if it exists.
    command = sys.argv[1]
    if command not in mapping.keys():
        show_help_and_exit(parser)
    else:
        mapping[command](arguments)


def execute_merge(arguments):
    import hypothesis.bin.io.merge


def execute_prune(arguments):
    import hypothesis.bin.io.prune


def execute_version(arguments):
    print(h.__version__)


def show_help_and_exit(parser):
    parser.print_help()
    sys.exit(0)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("merge", action="store_true", help="Test")
    parser.add_argument("version", action="store_true", help="Shows the version and commit of the hypothesis binary.")
    arguments, _ = parser.parse_known_args()

    return arguments


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
