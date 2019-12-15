r"""Utilities for hypothesis.nn."""

import hypothesis
import numpy as np
import torch



def compute_dimensionality(shape):
    dimensionality = 1
    for dim in shape:
        dimensionality *= dim

    return dimensionality
