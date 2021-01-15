r"""A utility program to merge data files.

"""

import argparse
import glob
import numpy as np
import os
import torch

from hypothesis.util.data.numpy import merge as numpy_merge



def main():
    arguments = parse_arguments()
    procedure = select_extension_procedure(arguments)
    procedure(arguments)


def procedure_numpy(arguments):
    files = fetch_input_files(arguments)
    numpy_merge(input_files=files,
        output_file=arguments.out,
        tempfile=arguments.tempfile,
        in_memory=arguments.in_memory,
        axis=arguments.dimension)


def procedure_torch(arguments):
    raise NotImplementedError


def fetch_input_files(arguments, delimiter=','):
    # Check if the user specified a list of input files
    if delimiter in arguments.files:
        files = arguments.files.split(delimiter)
    # Check if the specified file exists
    elif os.path.exists(arguments.files):
        files = [arguments.files]
    # The specified argument is a query
    else:
        query = arguments.files
        files = glob.glob(query)
    # Check if the list of files needs to be sorted.
    if arguments.sort:
        files.sort()

    return files


def select_extension_procedure(arguments):
    extension = arguments.extension
    mappings = {
        "numpy": procedure_numpy,
        "torch": procedure_torch}
    # Check if an extensions has been manually defined
    if extension in mappings.keys():
        procedure = mappings[extension]
    else:
        procecure = None

    return procedure


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=0, help="Axis in which to merge the data (default: 0).")
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
    # Check if an output path has been specified.
    if arguments.out is None:
        raise ValueError("No output path has been specified.")

    return arguments


if __name__ == "__main__":
    main()
