r"""Boilerplate file for defining your conditional ratio estimators."""

import hypothesis
import numpy as np
import torch

from hypothesis.nn import ConditionalDenseNetRatioEstimator
from hypothesis.nn import ConditionalMLPRatioEstimator
from hypothesis.nn import ConditionalResNetRatioEstimator



# General properties which are shared among your models.
activation_conv = torch.nn.ReLU
activation_trunk = torch.nn.ELU
trunk = (512, 512, 512)
shape_inputs = (1,) # CHANGEME
shape_outputs = (1,) # CHANGEME



class DummyRatioEstimator(ConditionalMLPRatioEstimator):

    def __init__(self,
        activation=activation_trunk,
        dropout=0.0,
        layers=trunk):
        super(DummyRatioEstimator, self).__init__(
            shape_inputs=shape_inputs,
            shape_outputs=shape_outputs,
            activation=activation,
            dropout=dropout,
            layers=layers)
        raise NotImplementedError("Dummy architecture, change me.")

    def log_ratio(self, inputs, outputs):
        # Maybe you want to do something special here.
        # For instance:
        inputs = inputs.log()

        return super(DummyRatioEstimator, self).log_ratio(inputs, outputs)



class AnotherDummyRatioEstimator(ConditionalResNetRatioEstimator):

    def __init__(self,
        depth=18,
        activation=activation_conv,
        batchnorm=True,
        dropout=0.0,
        in_planes=64,
        trunk=trunk,
        trunk_activation=activation_trunk,
        combined=False,
        log_inputs=False):
        super(AnotherDummyRatioEstimator, self).__init__(
            depth=depth,
            shape_inputs=shape_inputs,
            shape_outputs=shape_outputs,
            channels=1,
            activation=activation,
            batchnorm=batchnorm,
            trunk_activation=trunk_activation,
            trunk_dropout=dropout,
            in_planes=in_planes,
            trunk=trunk)

    def log_ratio(self, inputs, outputs):
        # Since this is a ResNet based ratio estimator, add a channel dimension.
        outputs = outputs.view(-1, 1, 99)

        return super(AnotherDummyRatioEstimator, self).log_ratio(inputs, outputs)
