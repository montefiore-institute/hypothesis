"""
Visualizations for MCMC algorithms.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt



def traceplot(result, **kwargs):
    # Argument key definitions.
    KEY_SHOW_BURNIN = "show_burnin"
    KEY_XLABEL = "xlabel"
    KEY_YLABEL = "ylabel"

    # Show argument defaults.
    show_burnin = False
    label_x = None
    label_y = None
    # Process optional arguments.
    if KEY_SHOW_BURNIN in kwargs.keys():
        show_burnin = bool(kwargs[KEY_SHOW_BURNIN])
    if KEY_XLABEL in kwargs.keys():
        label_x = str(kwargs[KEY_XLABEL])
    if KEY_YLABEL in kwargs.keys():
        label_y = str(kwargs[KEY_YLABEL])
    # Start the plotting procedure.


def autocorrelation(result, **kwargs):
    # Argument key definitions.
    KEY_SHOW_BURNIN = "show_burnin"
    KEY_XLABEL = "xlabel"
    KEY_YLABEL = "ylabel"

    # Set default argument values.
    show_burnin = False
    label_x = None
    label_y = None
    # Process optional arguments.
    raise NotImplementedError
