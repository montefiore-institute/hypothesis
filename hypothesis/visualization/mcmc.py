"""
Visualization mechanisms for Markov chain Monte Carlo.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from hypothesis.util.common import parse_argument



# Constants
KEY_RADIUS = "radius"
KEY_INTERVAL = "interval"
KEY_MAX_LAG = "max_lag"


def plot_autocorrelation(result, **kwargs):
    # Set the default arguments.
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
