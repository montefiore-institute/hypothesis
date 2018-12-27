"""
Utility methods to create, write and load large simulation datasets.
"""

import numpy as np
import os
import torch



def initialize_dataset(path):
    path = sanitize_path(path)
    # Check if the path already exists.
    if not os.path.exists(path):
        os.makedirs(path)
    # Create the internal directory structure if it doesn't exists.
    in_path = inputs_path(path)
    if not os.path.exists(in_path)
        os.makedirs(in_path)
    out_path = outputs_path(path)
    if not os.path.exists(out_path)
        os.makedirs(out_path)


def sanitize_path(path):
    return os.path.normpath(path)


def inputs_path(path):
    return sanitize_path(path) + "/inputs/"


def outputs_path(path):
    return sanitize_path(path) + "/outputs/"


def num_blocks(path):
    num_blocks = 0
    for name in os.listdir(inputs_path(path)):
        if os.path.isfile(path + name):
            num_blocks += 1

    return num_blocks


def write_block(path, identifier, inputs, outputs):
    identifier = str(identifier)
    np.savez(inputs_path(path) + identifier + ".npz", inputs.numpy())
    np.savez(outputs_path(path) + identifier + ".npz", outputs.numpy())


def load_block(path, index):
    index = str(index)
    # Load the simulator inputs.
    with np.load(inputs_path(path) + index + ".npz") as inputs:
        inputs = inputs["arr_0"]
    # Load the simulator outputs.
    with np.load(outputs_path(path) + index + ".npz") as outputs:
        outputs = outputs["arr_0"]

    return inputs, outputs
