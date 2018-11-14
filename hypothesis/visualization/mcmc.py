"""
Visualizations for MCMC algorithms.

All visualizations do NOT show the plots after the method has been called.
This gives the user the option to modify additonal parameters of the plot.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt



def plot_chains(result, **kwargs):
    raise NotImplementedError


def plot_densities(chains, **kwargs):
    raise NotImplementedError


def plot_density(result, **kwargs):
    # Argument key definitions.
    KEY_SHOW_MEAN = "show_mean"
    KEY_TRUTH = "truth"
    KEY_BINS = "bins"
    KEY_PARAMETER_INDEX = "parameter_index"

    # Set argument defaults.
    truth = None
    show_mean = False
    bins = 25
    parameter_index = 0
    # Process optional arguments.
    if KEY_TRUTH in kwargs.keys():
        truth = kwargs[KEY_TRUTH]
        if type(truth) is not list:
            truth = [truth]
    if KEY_SHOW_MEAN in kwargs.keys():
        show_mean = bool(kwargs[KEY_SHOW_MEAN])
    if KEY_BINS in kwargs.keys():
        bins = int(kwargs[KEY_BINS])
    if KEY_PARAMETER_INDEX in kwargs.keys():
        parameter_index = int(kwargs[KEY_PARAMETER_INDEX])
    # Start the plotting procedure.
    chain = result.chain(parameter_index)
    weights = [1. / result.size()] * result.size()
    plt.hist(chain.numpy(), bins=bins, weights=weights)
    plt.grid(True, alpha=.7)
    plt.minorticks_on()
    if truth:
         plt.axvline(truth[parameter_index], c='r', lw=2, linestyle='--', alpha=.95)
    if show_mean:
         plt.axvline(result.mean(parameter_index), c='y', lw=2, linestyle='--', alpha=.95)


def plot_trace(result, **kwargs):
    # Argument key definitions.
    KEY_SHOW_BURNIN = "show_burnin"
    KEY_TRUTH = "truth"
    KEY_ASPECT = "aspect"
    KEY_SHOW_MEAN = "show_mean"
    KEY_OFFSET = "offset"

    # Set argument defaults.
    show_burnin = False
    truth = None
    aspect = "auto"
    offset = .1
    show_mean = False
    # Process optional arguments.
    if KEY_SHOW_BURNIN in kwargs.keys():
        show_burnin = bool(kwargs[KEY_SHOW_BURNIN])
    if KEY_TRUTH in kwargs.keys():
        truth = kwargs[KEY_TRUTH]
        if type(truth) is not list:
            truth = [truth]
    if KEY_ASPECT in kwargs.keys():
        aspect = float(kwargs[KEY_ASPECT])
    if KEY_SHOW_MEAN in kwargs.keys():
        show_mean = bool(kwargs[KEY_SHOW_MEAN])
    if KEY_OFFSET in kwargs.keys():
        offset = float(kwargs[KEY_OFFSET])
    # Start the plotting procedure.
    max_iterations = result.iterations()
    num_parameters = result.parameters()
    if show_burnin and result.has_burnin():
        max_iterations += result.iterations(burnin=True)
    x = np.arange(1, max_iterations + 1)
    fig, ax = plt.subplots(nrows=num_parameters, ncols=1)

    def plot_chain(ax, parameter_index, aspect):
        if show_burnin and result.has_burnin():
            chain = torch.cat([result.chain(parameter_index, burnin=True), result.chain(parameter_index)], dim=0)
            ax.axvspan(0, result.iterations(burnin=True), alpha=0.25, color='gray')
        else:
            chain = result.chain(parameter_index)
        ax.grid(True, alpha=0.4)
        ax.set_xlim([0, len(chain)])
        ax.minorticks_on()
        ax.plot(x, chain.numpy(), alpha=.9)
        aspect = (1. / ax.get_data_ratio()) * (1. / aspect)
        ax.set_aspect(aspect)
        if not parameter_index:
            parameter_index = 0
        # Check if the truth has been specified.
        if truth:
            ax.axhline(truth[parameter_index], c='r', lw=2, linestyle='--', alpha=.95)
        # Set the y-limits of the plots.
        chain_mean = result.mean(parameter_index)
        min_chain = result.min()
        max_chain = result.max()
        delta_min = abs(min_chain - chain_mean)
        delta_max = abs(max_chain - chain_mean)
        if delta_min > delta_max:
            delta = delta_min
        else:
            delta = delta_max
        ax.set_ylim([chain_mean - delta - offset, chain_mean + delta + offset])
        # Check if the sample mean needs to be shown.
        if show_mean:
            ax.axhline(chain_mean, c='y', lw=2, linestyle='--', alpha=.95)

    if num_parameters > 1:
        for parameter_index, row in enumerate(ax):
            plot_chain(row, parameter_index, aspect)
    else:
        plot_chain(ax, None, aspect)

    return fig, ax


def plot_autocorrelation(result, **kwargs):
    # Argument key definitions.
    KEY_RADIUS = "radius"
    KEY_INTERVAL = "interval"
    KEY_MAX_LAG = "max_lag"

    # Set default arguments.
    radius = 1.1
    max_lag = None
    interval = 5
    center = .5
    # Parse the optional arguments.
    if KEY_RADIUS in kwargs.keys():
        radius = float(kwargs[KEY_RADIUS])
    if KEY_INTERVAL in kwargs.keys():
        interval = int(kwargs[KEY_INTERVAL])
    if KEY_MAX_LAG in kwargs.keys():
        max_lag = int(kwargs[KEY_MAX_LAG])
    # Compute the autocorrelation function.
    x, y = result.autocorrelation_function(max_lag, interval)
    # Start plotting the autocorrelation funtion.
    plt.ylim([-radius, radius])
    for index in range(len(x)):
        lag = x[index]
        autocorrelation = y[index]
        plt.axvline(lag, center, center + autocorrelation / 2 / radius, c="black")
    plt.minorticks_on()
    plt.grid(True, alpha=.7)
    plt.axhline(0, linestyle='--', c='r', alpha=.7, lw=1)
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
