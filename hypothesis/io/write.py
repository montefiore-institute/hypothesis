"""
Writing utilities.
"""

import h5py as h5
import numpy as np
import os
import torch



class writer:

    def __enter__(self):
        raise NotImplementedError

    def __exit__(self, *args):
        self.close()

    def close(self):
        raise NotImplementedError

    def insert(self, *args):
        raise NotImplementedError

    def flush(self):
        raise NotImplementedError
