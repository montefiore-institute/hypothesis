"""
Hypothesis is a python module for statistical inference.
"""

__version__ = "0.0.2"
__author__ = [
    "Joeri Hermans",
    "Volodimir Begy"
]
__email__ = "joeri.hermans@doct.uliege.be"

import torch

# Check the availability of a GPU.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



from .engine.hooks import call_hooks
from .engine.hooks import clear_hooks
from .engine.hooks import register_hook
from .engine import hooks
from .io import load
from .io import save
