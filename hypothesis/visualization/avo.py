"""
Visualization mechanisms for Adversarial Variational Optimization.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch

from hypothesis.util.common import parse_argument


# Constants
KEY_LABELX = "labelx"
KEY_LABELY = "labely"
KEY_SHOW_LABELS = "show_labels"
KEY_SHOW_MEAN = "show_mean"
KEY_TRUTH = "truth"


def plot_proposal(proposal, **kwargs):
    # Define the supported proposals.
    methods = {
        "NormalProposal": plot_normal_proposal,
        "MultivariateNormalProposal": plot_multivariate_normal_proposal}
    # Check if a method is available.
    key = type(proposal).__name__
    if key not in methods.keys():
        raise NotImplementedError("Proposal visualization not implemented for " + key)
    methods[key](proposal, **kwargs)


def plot_normal_proposal(proposal, **kwargs):
    # Fetch proposal parameters.
    mean = proposal.mu.item()
    sigma = proposal.sigma.item()
    # Check if the truth has been specified.
    truth = parse_argument(kwargs, key=KEY_TRUTH, default=None, type=float)
    show_mean = parse_argument(kwargs, key=KEY_SHOW_MEAN, default=False, type=bool)
    show_labels = parse_argument(kwargs, key=KEY_SHOW_LABELS, default=True, type=bool)
    labelx = parse_argument(kwargs, key=KEY_LABELX, default=r"$\theta$", type=str)
    labely = parse_argument(kwargs, key=KEY_LABELY, default=r"$q(\theta)$", type=str)
    if truth:
        plt.axvline(truth, c="red", lw=2, label=r"$\theta^*$")
    if show_mean:
        plt.axvline(mean, c="gray", lw=2, linestyle="--", alpha=.75)
    # Plot the proposal.
    x = np.linspace(mean - 10 * sigma, mean + 10 * sigma, 5000)
    plt.plot(x, scipy.stats.norm.pdf(x, mean, sigma))
    plt.xlim([mean - 6 * sigma, mean + 6 * sigma])
    if show_labels:
        plt.ylabel(labely)
        plt.xlabel(labelx)


def plot_multivariate_normal_proposal(proposal, **kwargs):
    raise NotImplementedError
