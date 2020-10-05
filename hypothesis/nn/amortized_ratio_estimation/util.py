import hypothesis
import numpy as np
import torch



def build_ratio_estimator(architecture, variables, **kwargs):
    creator = architectures[architecture]

    return creator(architecture, variables, **kwargs)


def build_mlp_ratio_estimator(architecture, variables, **kwargs):
    from hypothesis.nn.amortized_ratio_estimation.multi_layered_perceptron import build_ratio_estimator
    return build_ratio_estimator(variables)


def build_resnet_ratio_estimator(architecture, variables, **kwargs):
    raise NotImplementedError


def build_densenet_ratio_estimator(architecture, variables, **kwargs):
    raise NotImplementedError



architectures = {
    # Multi-Layered Perceptron
    "mlp": build_mlp_ratio_estimator,
    # ResNet
    "resnet": build_resnet_ratio_estimator,
    # DenseNet
    "densenet": build_densenet_ratio_estimator}
