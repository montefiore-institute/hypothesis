"""
Visualizations for MCMC algorithms.

All visualizations do NOT show the plots after the method has been called.
This gives the user the option to modify additonal parameters of the plot.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt



def plot_trace(result, **kwargs):
    # Argument key definitions.
    KEY_SHOW_BURNIN = "show_burnin"
    KEY_TRUTH = "truth"
    KEY_ASPECT = "aspect"
    KEY_SHOW_MEAN = "show_mean"
    KEY_OFFSET = "offset"

    # Show argument defaults.
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
    num_parameters = result.num_parameters()
    if show_burnin and result.has_burnin():
        max_iterations += result.burnin_iterations()
    x = np.arange(1, max_iterations + 1)
    fig, ax = plt.subplots(nrows=num_parameters, ncols=1)

    def plot_chain(ax, parameter_index, aspect):
        chain = []
        if show_burnin and result.has_burnin():
            chain = chain + result.burnin_chain(parameter_index)
            ax.axvspan(0, result.burnin_iterations(), alpha=0.25, color='gray')
        chain = chain + result.chain(parameter_index)
        ax.grid(True, alpha=0.4)
        ax.set_xlim([0, len(chain)])
        ax.minorticks_on()
        ax.plot(x, chain, alpha=.9)
        aspect = (1. / ax.get_data_ratio()) * (1. / aspect)
        ax.set_aspect(aspect)
        if not parameter_index:
            parameter_index = 0
        # Check if the truth has been specified.
        if truth:
            ax.axhline(truth[parameter_index], c='r', lw=2, linestyle='--', alpha=.95)
        # Set the y-limits of the plots.
        chain_mean = result.chain_mean(parameter_index)
        min_chain = result.chain_min()
        max_chain = result.chain_max()
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
    # Process optional arguments.
    raise NotImplementedError
