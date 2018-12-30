"""
Visualization mechanisms for Markov chain Monte Carlo.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from hypothesis.util.common import parse_argument



# Constants
KEY_BINS = "bins"
KEY_INTERVAL = "interval"
KEY_MAX_LAG = "max_lag"
KEY_PARAMETER_INDEX = "parameter_index"
KEY_RADIUS = "radius"
KEY_SHOW_MEAN = "show_mean"
KEY_TRUTH = "truth"


def plot_density(result, **kwargs):
    # Fetch the specified arguments, or their defaults.
    bins = get_argument(**kwargs, key=KEY_BINS, default=25, type=int)
    parameter_index = get_argument(**kwargs, key=KEY_PARAMETER_INDEX, default=0, type=int)
    show_mean = get_argument(**kwargs, key=KEY_SHOW_MEAN, default=False, type=bool)
    truth = get_argument(**kwargs, key=KEY_TRUTH, default=None, type=float)
    chain = result.chain(parameter_index=parameter_index)
    weights = [1 / result.size()] * result.size()
    plt.hist(chain.numpy(), bins=bins, weights=weights)
    plt.grid(True, alpha=.75)
    plt.minorticks_on()
    if truth:
        plt.axvline(truth, c='r', lw=2, linestyle="--",)
    if show_mean:
        plt.axvline(result.mean(parameter_index=parameter_index), c='y', lw=2, linestyle="--", alpha=.75)


def plot_autocorrelation(result, **kwargs):
    # Fetch the specified arguments, or their defaults.
    radius = get_argument(**kwargs, key=KEY_RADIUS, default=1.1, type=float)
    max_lag = get_argument(**kwargs, key=KEY_MAX_LAG, default=None, type=int)
    interval = get_argument(**kwargs, key=KEY_INTERVAL, default=5, type=int)
    # Compute the autocorrelation function.
    x, y = result.autocorrelation_function(max_lag=max_lag, interval=interval)
    plt.ylim([-radius, radius])
    # Plot the autocorrelation at the specified lags.
    for index in range(len(x)):
        lag = x[index]
        autocorrelation = y[index]
        plt.axvline(lag, center, center + autocorrelation / 2 / radius, c="black")
    plt.minorticks_on()
    plt.grid(True, alpha=.75)
    plt.axhline(0, linestyle="--", c='r', alpha=.75, lw=1)
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
