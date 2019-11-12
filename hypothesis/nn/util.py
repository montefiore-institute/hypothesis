r"""Utilities for hypothesis.nn."""

import numpy as np
import torch



def compute_output_shape(network, input_shape):
    inputs = torch.zeros(input_shape).unsqueeze(0)
    outputs = network(inputs)

    return outputs.shape
