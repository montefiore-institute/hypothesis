r"""A utility program to prune data files.

"""

import argparse
import numpy as np
import os
import torch



def main(arguments):
    raise NotImplementedError


def parse_arguments():
    parser = argparse.ArgumentParser("Prune: pruning data files for you convenience.")
    parser.add_argument("--dimension", type=int, default=1, help="Data dimension to work in (default: 1).")
    parser.add_argument("--in", type=str, default=None, help="Path to the file to process (default: none).")
    parser.add_argument("--in-memory", action="store_true", help="Process the data in-memory (default: false).")
    parser.add_argument("--indices", type=str, default=None, help="A comma-seperated list of indices to remove (default: none).")
    parser.add_argument("--out", type=str, default=None, help="Path of the processed file (default: none).")
    parser.add_argument("--tempfile", type=str, default=None, help="Path of the temporary file to store the intermediate results (default: none).")
    arguments, _ = parser.parse_known_args()
    # Check if an input file has been specified.
    if arguments.in is None:
        raise ValueError("No input file has been specified.")
    # Check if an output file has been specified.
    if arguments.out is None:
        raise ValueError("No output file has been specified.")

    return arguments


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
