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



class block_writer:

    def __init__(self, path, blocksize=10000):
        path = sanitize_path(path)
        self.blocksize = blocksize
        self.path = path
        self._reset()

    def _reset():
        self.block_index = 0
        self.block_inputs_buffer = []
        self.block_outputs_buffer = []

    def write(self, x, y):
        x = x.view(1, -1)
        y = y.view(1, -1)
        self.block_inputs_buffer.append(x)
        self.block_outputs_buffer.append(y)
        # Check if the block needs to be flushed.
        if len(self.block_inputs_buffer) == self.blocksize:
            self.flush()

    def flush(self):
        assert len(self.block_inputs_buffer) == self.blocksize
        inputs = torch.cat(self.block_inputs_buffer, dim=0)
        outputs = torch.cat(self.block_outputs_buffer, dim=0)
        write_block(self.path, self.block_index, inputs, outputs)
        self.block_index += 1

    def __enter__(self):
        self._reset()
        initialize_dataset(self.path)

    def __exit__(self, type, value, traceback):
        self.flush()
