r""""""

import matplotlib.pyplot as plt
import numpy as np
import torch

from hypothesis.visualization.util import make_square
from hypothesis.visualization.util import set_aspect



def plot_losses(losses_train, losses_test, epochs=None, log=True, figsize=None, ylim=None):
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
            epochs = np.arange(1, num_epochs + 1)
        # Allocate a figure with shared y-axes.
        figure, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize, sharey=True)
        plot_loss(axes[0], losses_train, epochs=epochs, title="Training loss",
            xlabel="Epochs", ylabel=ylabel)
        plot_loss(axes[1], losses_test, epochs=epochs, title="Test loss",
            xlabel="Epochs", ylabel=None)
        figure.tight_layout()
        if ylim is not None:
            axes[0].set_ylim(ylim)
            axes[1].set_ylim(ylim)
        make_square(axes[0])
        make_square(axes[1])

    return figure


def plot_loss(ax, losses, epochs=None, title=None, xlabel=None, ylabel=None):
    with torch.no_grad():
        # Check if the standard deviation needs to be computed.
        if losses.dim() == 2 and losses.shape[1] > 1:
            mean, std = losses.mean(dim=0), losses.std(dim=0)
            ax.plot(epochs, mean, color="black", lw=2, label="Loss")
            ax.fill_between(epochs, mean - std, mean + std, alpha=.25, color="black", label=r"$\pm1\sigma$ loss")
            ax.legend()
        else:
            ax.plot(epochs, losses.numpy(), color="black", lw=2)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.minorticks_on()
        make_square(ax)
