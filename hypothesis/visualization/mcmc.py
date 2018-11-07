"""
Visualizations for MCMC algorithms.

All visualizations do NOT show the plots after the method has been called.
This gives the user the option to modify additonal parameters of the plot.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt



def traceplot(result, **kwargs):
    # Argument key definitions.
    KEY_SHOW_BURNIN = "show_burnin"
    KEY_TRUTH = "truth"
    KEY_ASPECT = "aspect"

    # Show argument defaults.
    show_burnin = False
    truth = None
    aspect = .25
    # Process optional arguments.
    if KEY_SHOW_BURNIN in kwargs.keys():
        show_burnin = bool(kwargs[KEY_SHOW_BURNIN])
    if KEY_TRUTH in kwargs.keys():
        truth = kwargs[KEY_TRUTH]
        if type(truth) is not list:
            truth = [truth]
    if KEY_ASPECT in kwargs.keys():
        aspect = float(kwargs[KEY_ASPECT])
    # Start the plotting procedure.
    max_iterations = result.iterations()
    num_parameters = result.num_parameters()
    if show_burnin and result.has_burnin():
        max_iterations += result.burnin_iterations()
    x = np.arange(1, max_iterations + 1)
    fig, ax = plt.subplots(nrows=num_parameters, ncols=1)

    def plot_chain(ax, parameter_index):
        chain = []
        if show_burnin:
            chain = chain + result.burnin_chain(parameter_index)
        chain = chain + result.chain(parameter_index)
        ax.set_aspect(aspect)
        print(aspect)
        ax.plot(x, chain)
        if not parameter_index:
            parameter_index = 0
        ax.axhline(truth[parameter_index], c='r', lw=2, linestyle='--', alpha=.7)

    if num_parameters > 1:
        for parameter_index, row in enumerate(ax):
            plot_chain(row, parameter_index)
    else:
        plot_chain(ax, None)

    return fig, ax


def autocorrelation(result, **kwargs):
    # Argument key definitions.
    KEY_SHOW_BURNIN = "show_burnin"

    # Set default argument values.
    show_burnin = False
    label_x = None
    label_y = None
    # Process optional arguments.
    raise NotImplementedError
