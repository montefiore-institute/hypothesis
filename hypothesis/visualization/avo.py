"""
Visualizations for AVO.

All visualizations do NOT show the plots after the method has been called.
This gives the user the option to modify additional parameters.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt



def plot_proposal(proposal, **kwargs):
    # Argument key definitions.
    KEY_TRUTH = "truth"

    # Set argument defaults.
    truth = None
    # Parse optional arguments.
    if KEY_TRUTH in kwargs.keys():
        truth = float(kwargs[KEY_TRUTH])
    # Start plotting procedure.
    raise NotImplementedError
