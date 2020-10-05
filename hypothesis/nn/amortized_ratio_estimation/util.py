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
    from hypothesis.nn.amortized_ratio_estimation.resnet import  build_ratio_estimator
    return build_ratio_estimator(variables, **kwargs)


def build_resnet_with_depth_ratio_estimator(architecture, variables, **kwargs):
    _, depth = architecture.split('-')
    kwargs["depth"] = depth
    return build_resnet_ratio_estimator(architecture, variables, **kwargs)


def build_densenet_ratio_estimator(architecture, variables, **kwargs):
    from hypothesis.nn.amortized_ratio_estimation.densenet import build_ratio_estimator
    return build_ratio_estimator(variables, **kwargs)


def build_densenet_with_depth_ratio_estimator(architecture, variables, **kwargs):
    raise NotImplementedError


architectures = {
    # Multi-Layered Perceptron
    "mlp": build_mlp_ratio_estimator,
    # ResNet
    "resnet": build_resnet_ratio_estimator,
    "resnet-18": build_resnet_with_depth_ratio_estimator,
    "resnet-34": build_resnet_with_depth_ratio_estimator,
    "resnet-50": build_resnet_with_depth_ratio_estimator,
    "resnet-101": build_resnet_with_depth_ratio_estimator,
    "resnet-152": build_resnet_with_depth_ratio_estimator,
    # DenseNet
    "densenet": build_densenet_ratio_estimator,
    "densenet-121": build_densenet_with_depth_ratio_estimator,
    "densenet-161": build_densenet_with_depth_ratio_estimator,
    "densenet-169": build_densenet_with_depth_ratio_estimator,
    "densenet-201": build_densenet_with_depth_ratio_estimator}
