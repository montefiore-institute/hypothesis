r"""Utilities for ratio estimators.

"""

import hypothesis as h
import torch

from hypothesis.exception import UnknownRatioEstimatorError


def build_ratio_estimator(architecture, random_variables, **kwargs):
    r"""Utility method to easily create ratio estimator
    for various problem domains.

    """
