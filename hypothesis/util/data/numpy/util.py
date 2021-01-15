import glob
import numpy as np
import os
import tempfile as temp
import torch



def compute_final_shape(file_names, axis=0):
    num_files = len(file_names)
    shape = list(np.load(file_names[0]).shape)
    shape[axis] *= num_files

    return tuple(shape)


def merge(input_files, output_file, tempfile=None, dtype=None, in_memory=False, axis=0):
    # Compute the shape of the final data file.
    shape = compute_final_shape(input_files, axis=axis)
    # Check if a dtype needs to be derived.
    if dtype is None:
        dtype = np.load(input_files[0]).dtype
    if in_memory:
        merge_in_memory(input_files, output_file, shape=shape, dtype=dtype, axis=axis)
    else:
        merge_on_disk(input_files, output_file, shape=shape, dtype=dtype, axis=axis, tempfile=tempfile)


def merge_in_memory(input_files, output_file, shape, dtype=None, axis=0):
    datamap = np.zeros(shape, dtype=dtype)
    insert_data(input_files, datamap, axis=axis)
    np.save(output_file, datamap)


def merge_on_disk(input_files, output_file, shape, dtype=None, axis=0, tempfile=None):
    # Check if a random temporary file needs to be allocated.
    if tempfile is None:
        _, tempfile = temp.mkstemp(dir='.')
    datamap = np.memmap(tempfile, dtype=dtype, mode="w+", shape=shape)
    insert_data(input_files, datamap, axis=axis)
    np.save(output_file, datamap)
    os.remove(tempfile)


def insert_data(input_files, datamap, axis=0):
    index = 0
    if axis > 0:
        datamap = np.rollaxis(datamap, axis)
    for file_name in input_files:
        data = np.load(file_name)
        num_rows = data.shape[0]
        datamap[index:index + num_rows, :] = data
        index += num_rows
