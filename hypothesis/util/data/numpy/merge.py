import glob
import numpy as np
import os
import tempfile
import torch



def compute_final_shape(file_names):
    num_files = len(file_names)
    shape = list(np.load(file_names[0])).shape
    shape[0] *= num_files

    return tuple(shape)


def merge(pattern, output_file, tempfile=None, dtype=np.float32):
    file_names = glob.glob(pattern)
    shape = compute_final_shape(file_names)
    if tempfile is None:
        _, tempfile = tempfile.pkstemp()
    data_map = np.memmap(tempfile, dtype=dtype, mode="w+", shape=shape)
    index = 0
    for file_name in file_names:
        data = np.load(file_name)
        shape = data.shape
        rows = shape[0]
        data_map[index:index + rows, :] = data
        index += rows
    os.remove(tempfile)
