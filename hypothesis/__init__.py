"""
`hypothesis` is a python module for inference in settings were the likelihood is
intractable or unatainable.
"""

__version__ = "0.0.1"
__author__ = [
    "Joeri Hermans",
    "Volodimir Begy"
]
__email__ = "joeri.hermans@doct.uliege.be"



from .io import load
from .io import save
from .engine.hooks import call_hooks
from .engine.hooks import clear_hooks
from .engine.hooks import register_hook
from .engine import hooks
