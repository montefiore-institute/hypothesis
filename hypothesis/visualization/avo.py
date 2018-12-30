"""
Visualization mechanisms for Adversarial Variational Optimization.
"""

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import torch


# Constants
KEY_TRUTH = "truth"
KEY_SHOW_MEAN = "show_mean"


def plot_proposal(proposal, **kwargs):
    # Define the supported proposals.
    methods = {
        "NormalProposal": plot_normal_proposal,
        "MultivariateNormalProposal": plot_multivariate_normal_proposal}
    # Check if a method is available.
    key = proposal.__name__
    if key not in methods.keys():
        raise NotImplementedError("Proposal visualization not implemented for " + key)
    methods[key](proposal, **kwargs)


def plot_normal_proposal(proposal, **kwargs):
    # Fetch proposal parameters.
    mean = proposal.mu.item()
    sigma = proposal.sigma.item()
    # Check if the truth has been specified.
    if KEY_TRUTH in kwargs.keys():
        truth = float(kwargs[KEY_TRUTH])
        plt.axvline(truth, c="red", lw=2, label=r"$\theta^*$")
    # Check if the mean of the proposal needs to be plotted.
    if KEY_SHOW_MEAN in kwargs.keys():
        show_mean = bool(kwargs[KEY_SHOW_MEAN])
        if show_mean:
            plt.axvline(mean, c="gray", lw=2, style="--", alpha=.75, label="Proposal mean")
    # Plot the proposal.
    x = np.linspace(mean - 10 * sigma, mean + 10 * sigma, 5000)
    plt.plot(x, mlab.normpdf(x, mean, sigma), label=r"$q({\theta}|{\psi})\ \gamma = 0$")
    plt.xlim([mean - 6 * sigma, mean + 6 * sigma])
    plt.legend()


def plot_multivariate_normal_proposal(proposal, **kwargs):
    raise NotImplementedError
