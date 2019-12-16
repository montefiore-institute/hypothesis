r"""Utilities for hypothesis.nn."""

import hypothesis
import numpy as np
import torch

from hypothesis.util import is_iterable



def allocate_output_transform(transformation, shape):
    mapping = None
    if is_iterable(shape):
        dimensionality = compute_dimensionality(shape)
    else:
        dimensionality = shape
    if transformation is "normalize":
        if dimensionality > 1:
            mapping = torch.nn.Softmax(dim=0)
        else:
            mapping = torch.nn.Sigmoid()
    elif transformation is not None:
        mapping = transformation()

    return mapping


def compute_dimensionality(shape):
    dimensionality = 1
    for dim in shape:
        dimensionality *= dim

    return dimensionality
