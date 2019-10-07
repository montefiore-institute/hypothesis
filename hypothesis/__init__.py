r"""Hypothesis is a python module for statistical inference.

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
# Load PyTorch, and set related global variables.
################################################################################

import torch



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
