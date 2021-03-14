r"""Utilities for ratio estimators.

"""

import hypothesis as h
import torch

from hypothesis.exception import UnknownRatioEstimatorError


def build_ratio_estimator(architecture, random_variables, denominator="inputs|outputs", **kwargs):
    r"""Utility method to easily create ratio estimator
    for various problem domains.

    The following types are currently supported:
     - `mlp`
     - `resnet`, equivalent to `resnet-18`.
     - `resnet-18`
     - `resnet-34`
     - `resnet-50`
     - `resnet-101`
     - `resnet-152`
     - `densenet`, equivalent to `densenet-121`.
     - `densenet-121`
     - `densenet-161`
     - `densenet-169`
     - `densenet-201`

    The `random_variables` argument should be a dictionary which
    corresponds to the name of the random variable and it's data
    shape (without the batch-size).
    """
    if not architecture in architectures.keys():
        raise UnknownRatioEstimatorError
    creator = _architectures[architecture]

    return creator(variables, **kwargs)


def build_mlp_ratio_estimator(random_variables, denominator, **kwargs):
    r"""Utility method to easily create MLP-based ratio estimators.
    """
    from hypothesis.nn.ratio_estimation.mlp import build_ratio_estimator
    return build_ratio_estimator(random_variables, denominator, **kwargs)


_architectures = {
    # Multi-Layered Perceptron
    "mlp": build_mlp_ratio_estimator,
    # ResNet
    "resnet": build_resnet_18_ratio_estimator,
    "resnet-18": build_resnet_18_ratio_estimator}
