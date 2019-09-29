"""Hypothesis is a python module for statistical inference.

The module contains (approximate) inference algorithms with PyTorch
integration. Additionally, utilities are provided for data loading, efficient
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


"""str: String describing the PyTorch accelerator backend.

The variable will be initialized when Hypothesis is loaded. It will check for
the availibility of CUDA. If a CUDA enabled device is present, it will
select the CUDA device defined in the `CUDA_VISIBLE_DEVICES` environment
variable. If no such device is specified, it will default to GPU 0.
"""
accelerator = torch.device("cuda" if torch.cuda.is_available() else "cpu")
