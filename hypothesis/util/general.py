r"""General utilities for Hypothesis."""

import hypothesis
import numpy as np
import torch



def is_iterable(item):
    return hasattr(item, "__getitem__")
