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
