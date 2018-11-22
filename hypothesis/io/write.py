"""
Writing utilities.
"""

import h5py as h5
import numpy as np
import torch



class Writer:

    def __enter__(self):
        raise NotImplementedError

    def __exit__(self, *kwargs):
        raise NotImplementedError

    def append(self, **kwargs):
        raise NotImplementedError

    def flush(self):
        raise NotImplementedError
