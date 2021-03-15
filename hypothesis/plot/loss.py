r"""Plotting utilities for plotting loss curves.

"""

import glob
import matplotlib.pyplot as plt
import numpy as np
import torch

from .util import *


def loss(files, ax=None, variance=True, **kwargs):
    r"""Plots a single, averaged, loss curves based on the data files provided.
    The specified `files` argument can be an `glob` pattern. In addition,
    whenever no custom `matplotlib` axes has been specified through `ax`, the
    method will default to `plt.gca()`.

    Additional arguments for `ax.plot` can be specified through the `kwargs` argument.

    For example, imagine you have trained various models;

        import hypothesis as h

        # If you want to use the Hypothesis plotting style
        h.plot.activate()

        # Shows the variance between the losses by default.
        pattern = "dir/model-*/losses-test.npy"
        h.plot.loss(pattern, label="My model")
        plt.legend()
        plt.show()

        # Without the variances
        h.plot.loss(pattern, variance=False)
        plt.show()
    """
    # Check if a specific Matplotlib axes has been specified.
    if ax is None:
        ax = plt.gca()
    # Default theming.
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    # Generate the plot.
    data_files = glob.glob(files)
    loss_curves = []
    for file in data_files:
        loss_curves.append(np.load(file).reshape(1, -1))
    loss_curves = np.vstack(loss_curves)
    best = np.min(loss_curves, axis=0)
    m = np.mean(loss_curves, axis=0)
    epochs = np.arange(len(m)) + 1
    p = ax.plot(epochs, best, **kwargs)
    if len(data_files) >= 2 and variance:
        s = np.std(loss_curves, axis=0)
        ax.fill_between(epochs, m + s, m - s, color=p[0].get_color(), alpha=.1)
