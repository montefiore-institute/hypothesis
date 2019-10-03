r""""""

import corner
import matplotlib.pyplot as plt
import numpy as np
import torch

from hypothesis.summary.mcmc import Chain
from hypothesis.visualization.util import make_square
from hypothesis.visualization.util import set_aspect



def plot_autocorrelation(chain, parameter_index=None):
    raise NotImplementedError


def plot_density(chain):
    raise NotImplementedError


def plot_trace(chain, parameter_index=None):
    nrows = chain.dimensionality()[0]
    figure, rows = plt.subplots(nrows, 2, sharey=False, sharex=False, figsize=(5, 5))
    num_samples = chain.size()
    def display(ax_trace, ax_density, theta_index=1):
        # Trace
        ax_trace.minorticks_on()
        ax_trace.plot(range(num_samples), chain.samples.numpy(), color="black", lw=2)
        ax_trace.set_xlim([0, num_samples])
        ax_trace.set_xticks([])
        ax_trace.set_ylabel(r"$\theta_" + str(theta_index) + "$")
        limits = ax_trace.get_ylim()
        # Density
        ax_density.minorticks_on()
        ax_density.hist(chain.samples.numpy(), bins=50, lw=2, color="black", histtype="step", density=True)
        ax_density.yaxis.tick_right()
        ax_density.yaxis.set_label_position("right")
        ax_density.set_ylabel("Probability mass function")
        ax_density.set_xlabel(r"$\theta_" + str(theta_index) + "$")
        ax_density.set_xlim(limits)
        # Aspects
        make_square(ax_density)
        set_aspect(ax_trace, 1)
        make_square(ax_density)
    if nrows > 1:
        for index, ax_trace, ax_density in enumerate(rows):
            display(ax_trace, ax_density)
    else:
        ax_trace, ax_density = rows
        display(ax_trace, ax_density)

    return figure
