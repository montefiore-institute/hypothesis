r"""Utilities to obtain Bayesian credible regions or
Frequentist confidence intervals.

"""

import math
import numpy as np
import torch

from scipy.stats import chi2


@torch.no_grad()
def likelihood_ratio_test_statistic(log_ratios):
    max_ratio = log_ratios[log_ratios.argmax()]
    test_statistic = -2 * (log_ratios - max_ratio)
    test_statistic -= test_statistic.min()

    return test_statistic


@torch.no_grad()
def confidence_level(log_ratios, dof=None, level=0.95):
    if dof is None:
        dof = log_ratios.dim() - 1
    test_statistic = likelihood_ratio_test_statistic(log_ratios)
    level = chi2.isf(1 - level, df=dof)

    return test_statistic, level


@torch.no_grad()
def highest_density_level(density, alpha, min_epsilon=10e-16, region=False):
    # Check if a numpy type has been specified
    if type(density).__module__ != np.__name__:
        density = density.cpu().clone().numpy()
    else:
        density = np.array(density)
    density = density.astype(np.float64)
    # Check the discrete sum of the density (for scaling)
    integrand = density.sum()
    density /= integrand
    # Compute the level such that 1 - alpha has been satisfied.
    optimal_level = density.max()
    epsilon = 10e-00  # Current error
    while epsilon >= min_epsilon:
        optimal_level += 2 * epsilon  # Overshoot solution, move back
        epsilon /= 10
        area = 0.0
        while area < (1 - alpha):
            area_under = (density >= optimal_level)
            area = np.sum(area_under * density)
            optimal_level -= epsilon  # Gradient descent to reduce error
    # Rescale to original
    optimal_level *= integrand
    # Check if the computed mask needs to be returned
    if region:
        return optimal_level, area_under
    else:
        return optimal_level
