r"""Hypothesis is a python module for statistical inference and the
mechanization of science.

The package contains (approximate) inference algorithms to solve statistical
problems. Utilities are provided for data loading, efficient
simulation, visualization, fire-and-forget inference, and validation.
"""

__version__ = "0.0.3"
__author__ = [
    "Joeri Hermans"]

__email__ = [
    "joeri.hermans@doct.uliege.be"]


################################################################################
# Global variables
################################################################################

import multiprocessing
import torch



cpu_count = multiprocessing.cpu_count()
"""int: Number of available processor cores.

Variable will be initialized when ``hypothesis`` is loaded fro the first time.
"""



workers = cpu_count
"""int: Number of default workers.

Default number of workers in Hypothesis.
"""


def set_workers(n):
    r"""Sets the number of default hypothesis workers."""
    assert(n >= 1)
    hypothesis.workers = n



accelerator = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""torch.device: PyTorch device describing the accelerator backend.

The variable will be initialized when ``hypothesis`` is loaded for the first
time. It will check for the availibility of a CUDA device. If a CUDA enabled
device is present, ``hypothesis`` will select the CUDA device defined in the
``CUDA_VISIBLE_DEVICES`` environment variable. If no such device is specified,
the variable will default to GPU 0.
"""


def disable_gpu():
    r"""Disables GPU acceleration. Hypothesis' accelerator will have been
    set to 'cpu'."""
    hypothesis.accelerator = "cpu"

def enable_gpu():
    r"""Tries to enable GPU acceleration. If a GPU is present, a CUDA
    device will be set, else it will default to 'cpu'."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gpu_available():
    r"""Checks if GPU acceleration is available."""
    return hypothesis.accelerator is not "cpu"


################################################################################
# Hypothesis' defaults
################################################################################

import hypothesis.default
