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
KEY_SHOW_BURNIN = "show_burnin"
KEY_SHOW_MEAN = "show_mean"
KEY_TRUTH = "truth"
KEY_OFFSET = "offset"


def plot_density(result, **kwargs):
    # Fetch the specified arguments, or their defaults.
    bins = parse_argument(kwargs, key=KEY_BINS, default=25, type=int)
    parameter_index = parse_argument(kwargs, key=KEY_PARAMETER_INDEX, default=0, type=int)
    show_mean = parse_argument(kwargs, key=KEY_SHOW_MEAN, default=False, type=bool)
    truth = parse_argument(kwargs, key=KEY_TRUTH, default=None, type=float)
    chain = result.get_chain(parameter_index=parameter_index)
    weights = [1 / result.size()] * result.size()
    plt.hist(chain.numpy(), bins=bins, weights=weights)
    plt.grid(True, alpha=.75)
    plt.minorticks_on()
    if truth:
        plt.axvline(truth, c='r', lw=2, linestyle="--",)
    if show_mean:
        plt.axvline(result.mean(parameter_index=parameter_index), c='y', lw=2, linestyle="--", alpha=.75)


def plot_trace(result, **kwargs):
    # Fetch the specified arguments, or their defaults.
    parameter_index = parse_argument(kwargs, key=KEY_PARAMETER_INDEX, default=0, type=int)
    show_burnin = parse_argument(kwargs, key=KEY_SHOW_BURNIN, default=False, type=bool)
    show_mean = parse_argument(kwargs, key=KEY_SHOW_MEAN, default=True, type=bool)
    truth = parse_argument(kwargs, key=KEY_TRUTH, default=None, type=float)
    offset = parse_argument(kwargs, key=KEY_OFFSET, default=.1, type=float)
    # Start the plotting procedure.
    max_iterations = result.size()
    chain = result.get_chain(parameter_index=parameter_index, burnin=False)
    min_chain = chain.min().item()
    max_chain = chain.max().item()
    # Plot the burnin, if requested and available.
    if show_burnin and result.has_burnin():
        burnin_iterations = result.size(burnin=True)
        max_iterations += burnin_iterations
        burnin_chain = result.get_chain(parameter_index=parameter_index, burnin=True)
        chain = torch.cat([burnin_chain, chain], dim=0)
        plt.axvspan(0, burnin_iterations, alpha=.25, color="gray")
    # Check if the mean needs to be displayed.
    if show_mean:
        chain_mean = chain.mean(dim=0)
        plt.axhline(chain_mean, c='y', linestyle="--", lw=2, alpha=.95, zorder=10)
    x = np.arange(1, max_iterations + 1)
    plt.grid(True, alpha=.4)
    plt.minorticks_on()
    plt.plot(x, chain.numpy(), alpha=.9)
    plt.xlim([0, max_iterations])
    plt.ylim([min_chain - offset, max_chain + offset])


def plot_autocorrelation(result, **kwargs):
    # Fetch the specified arguments, or their defaults.
    parameter_index = parse_argument(kwargs, key=KEY_PARAMETER_INDEX, default=0, type=int)
    radius = parse_argument(kwargs, key=KEY_RADIUS, default=1.1, type=float)
    max_lag = parse_argument(kwargs, key=KEY_MAX_LAG, default=None, type=int)
    interval = parse_argument(kwargs, key=KEY_INTERVAL, default=5, type=int)
    # Compute the autocorrelation function.
    x, y = result.autocorrelation_function(max_lag=max_lag, interval=interval, parameter_index=parameter_index)
    plt.ylim([-radius, radius])
    # Plot the autocorrelation at the specified lags.
    for index in range(len(x)):
        lag = x[index]
        autocorrelation = y[index]
        center = .5
        plt.axvline(lag, center, center + autocorrelation / 2 / radius, c="black")
    plt.minorticks_on()
    plt.grid(True, alpha=.75)
    plt.axhline(0, linestyle="--", c='r', alpha=.75, lw=1)
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
