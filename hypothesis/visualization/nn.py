r""""""

import matplotlib.pyplot as plt
import numpy as np
import torch

from hypothesis.visualization.util import make_square
from hypothesis.visualization.util import set_aspect



def plot_losses(losses_train, losses_test, epochs=None, log=True, figsize=None):
    with torch.no_grad():
        # Check if the losses have to be converted to log-space.
        if log:
            ylabel = "Logarithmic loss"
            losses_train = losses_train.log()
            losses_test = losses_test.log()
        else:
            ylabel = "Loss"
        # Check if custom epochs have been specified.
        if epochs is None:
            if losses_train.dim() == 2:
                num_epochs = losses_train.shape[1]
            else:
                num_epochs = losses_train.shape[0]
            epochs = np.arange(1, num_epochs)
        # Allocate a figure with shared y-axes.
        figure, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize, sharey=True)
        plot_loss(axes[0], losses_train, epochs=epochs, title="Training loss",
            xlabel="Epochs", ylabel=ylabel, log=log)
        plot_loss(axes[1], losses_test, epochs=epochs, title="Test loss",
            xlabel="Epochs", ylabel=ylabel, log=log)
        figure.tight_layout()

    return figure


def plot_loss(ax, losses, epochs=None, title=None, xlabel=None, ylabel=None):
    with torch.no_grad():
        # Check if the standard deviation needs to be computed.
        mean = losses.mean(dim=)
        ax.legend()
        make_square(ax)
