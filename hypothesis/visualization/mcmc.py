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


def plot_autocorrelation(chain, interval=2, max_lag=100, radius=1.1):
    if max_lag is None:
        max_lag = chain.size()
    autocorrelations = chain.autocorrelations()[:max_lag]
    lags = np.arange(0, max_lag, interval)
    autocorrelations = autocorrelations[lags]
    plt.ylim([-radius, radius])
    center = .5
    for index, lag in enumerate(lags):
        autocorrelation = autocorrelations[index]
        plt.axvline(lag, center, center + autocorrelation / 2 / radius, c="black")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.minorticks_on()
    plt.axhline(0, linestyle="--", c="black", alpha=.75, lw=2)
    make_square(plt.gca())
    figure = plt.gcf()

    return figure


def plot_trace(chain, parameter_index=None):
    nrows = chain.dimensionality()
    figure, rows = plt.subplots(nrows, 2, sharey=False, sharex=False, figsize=(2 * 7, 2))
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
        ax_trace.set_aspect("auto")
        ax_trace.set_position([0, 0, .7, 1])
        ax_density.set_position([.28, 0, 1, 1])
    if nrows > 1:
        for index, ax_trace, ax_density in enumerate(rows):
            display(ax_trace, ax_density)
    else:
        ax_trace, ax_density = rows
        display(ax_trace, ax_density)

    return figure
