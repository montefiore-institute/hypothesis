r"""A utility program to prune data files.

"""

import argparse
import numpy as np
import os
import shutil
import torch



def main(arguments):
    # Check if the file needs to be processed in memory.
    if arguments.in_memory:
        mmap_mode = 'r'
    else:
        mmap_mode = None
    data = np.load(arguments.in_file, mmap_mode=mmap_mode)
    data = np.delete(data, arguments.indices, arguments.dimension)
    np.save(arguments.out_file, data)


def parse_arguments():
    parser = argparse.ArgumentParser("Prune: pruning data files for you convenience.")
    parser.add_argument("--dimension", type=int, default=1, help="Data dimension to work in (default: 1).")
    parser.add_argument("--in-file", type=str, default=None, help="Path to the file to process (default: none).")
    parser.add_argument("--in-memory", action="store_true", help="Process the data in-memory (default: false).")
    parser.add_argument("--indices", type=str, default=None, help="A comma-seperated list of indices to remove (default: none).")
    parser.add_argument("--out-file", type=str, default=None, help="Path of the processed file (default: none).")
    arguments, _ = parser.parse_known_args()
    # Check if an input file has been specified.
    if arguments.in_file is None:
        raise ValueError("No input file has been specified.")
    # Check if an output file has been specified.
    if arguments.out_file is None:
        raise ValueError("No output file has been specified.")
    # Check if indices have been specified.
    if arguments.indices is None:
        raise ValueError("No indices have been specified.")
    arguments.indices = [int(index) for index in arguments.indices.split(',')]

    return arguments


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
