r"""Utilities for hypothesis.nn."""

import numpy as np
import torch



def compute_output_dimensionality(network, input_shape):
    dimensionality = 1
    inputs = torch.zeros(input_shape).unsqueeze(0)
    outputs = network(inputs)
    for element in outputs.shape:
        dimensionality *= element

    return dimensionality
