r"""A utility program to merge data files.

Use-cases include merging batch simulations.
"""

import argparse
import glob
import numpy as np
import os
import torch

from hypothesis.util.data.numpy import merge as numpy_merge



def main(arguments):
    procedure = select_extension_procedure(arguments)
    procedure(arguments)


def procedure_numpy(arguments):
    tempfile = str(uuid.uuid4())
    raise NotImplementedError


def procedure_numpy_in_memory(arguments):
    raise NotImplementedError


def procedure_torch(arguments):
    raise NotImplementedError


def procedure_torch_in_memory(arguments):
    raise NotImplementedError


def select_extension_procedure(arguments):
    extension = arguments.extension
    if arguments.in_memory:
        extension += "-in-memory"
    mappings = {
        "numpy": procedure_numpy,
        "numpy-in-memory": procedure_numpy_in_memory,
        "torch": procedure_torch,x
        "torch-in-memory": procedure_torch_in_memory}
    if extension in mappings.keys():
        procedure = mappings[extension]
    else:
        procedure = None

    return procedure


def parse_arguments():
    parser = argparse.ArgumentParser("Merge: merging data files for your convenience.")
    parser.add_argument("--dimension", type=int, default=0, help="Dimension in which to merge the data (default: 0).")
    parser.add_argument("--extension", type=str, default=None, help="Data file to process, available options: numpy, torch. (default: none).")
    parser.add_argument("--files", type=str, default=None, help="A list of files delimited by ',' or a glob pattern (default: none).")
    parser.add_argument("--in-memory", action="store_true", help="Processes all chunks in memory (default: false).")
    parser.add_argument("--out", type=str, default=None, help="Output path to store the result (default: none).")
    parser.add_argument("--sort", action="store_true", help="Sort the input files before processing (default: false).")
    parser.add_argument("--tempfile", type=str, default=None, help="Path of the temporary file to store the intermediate results, only accessible to non in-memory operations. (default: none).")
    arguments, _ = parser.parse_known_args()
    # Check if a proper extension has been specified.
    if select_extension_procedure(arguments) is None:
        raise ValueError("The specified extention (", arguments.extension, ") does not exists.")
    # Check if files to merge have been specified.
    if arguments.files is None:
        raise ValueError("No input files have been specified.")
    # Check if a list of files has been specified.
    if ',' in arguments.files:
        arguments.files = arguments.files.split(',')
    else:
        arguments.files = glob.glob(arguments.files)
    # Check if the files have to be sorted.
    if arguments.sort:
        arguments.files.sort()
    # Check if an output path has been specified.
    if arguments.out is None:
        raise ValueError("No output path has been specified.")

    return arguments


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
