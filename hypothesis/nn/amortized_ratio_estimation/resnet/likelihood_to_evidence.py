import hypothesis
import hypothesis.nn
import torch

from hypothesis.nn import MultiLayeredPerceptron
from hypothesis.nn import ResNetHead
from hypothesis.nn.amortized_ratio_estimation import BaseLikelihoodToEvidenceRatioEstimator
from hypothesis.nn.resnet.default import batchnorm as default_batchnorm
from hypothesis.nn.resnet.default import channels as default_channels
from hypothesis.nn.resnet.default import convolution_bias as default_convolution_bias
from hypothesis.nn.resnet.default import depth as default_depth
from hypothesis.nn.resnet.default import dilate as default_dilate
from hypothesis.nn.resnet.default import groups as default_groups
from hypothesis.nn.resnet.default import in_planes as default_in_planes
from hypothesis.nn.resnet.default import width_per_group as default_width_per_group
from hypothesis.nn.util import compute_dimensionality



class LikelihoodToEvidenceRatioEstimatorResNet(BaseLikelihoodToEvidenceRatioEstimator):

    def __init__(self,
        shape_inputs,
        shape_outputs,
        activation=hypothesis.default.activation,
        batchnorm=default_batchnorm,
        channels=default_channels,
        convolution_bias=default_convolution_bias,
        depth=default_depth,
        dilate=default_dilate,
        groups=default_groups,
        in_planes=default_in_planes,
        width_per_group=default_width_per_group,
        trunk_activation=None,
        trunk_dropout=hypothesis.default.dropout,
        trunk_layers=hypothesis.default.trunk):
        super(LikelihoodToEvidenceRatioEstimatorResNet, self).__init__()
        # Construct the convolutional ResNet head.
        self.head = ResNetHead(
            activation=hypothesis.default.activation,
            batchnorm=batchnorm,
            channels=channels,
            convolution_bias=convolution_bias,
            depth=depth,
            dilate=dilate,
            groups=groups,
            in_planes=in_planes,
            shape_xs=shape_xs,
            width_per_group=width_per_group)
        # Check if custom trunk settings have been defined.
        if trunk_activation is None:
            trunk_activation = activation
        # Construct the trunk of the network.
        dimensionality = self.head.embedding_dimensionality() + compute_dimensionality(shape_inputs)
        self.trunk = MLP(
            shape_xs=(dimensionality,),
            shape_ys=(1,),
            activation=trunk_activation,
            dropout=trunk_dropout,
            layers=trunk_layers,
            transform_output=None)

    def log_ratio(self, inputs, outputs):
        z_head = self.head(outputs)
        features = torch.cat([inputs, z_head])

        return self.trunk(features)
