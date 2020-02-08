import hypothesis
import hypothesis.nn.resnet
import torch

from .default import batchnorm as default_batchnorm
from .default import channels as default_channels
from .default import convolution_bias as default_convolution_bias
from .default import depth as default_depth
from .default import dilate as default_dilate
from .default import groups as default_groups
from .default import in_planes as default_in_planes
from .default import width_per_group as default_width_per_group
from hypothesis.nn import MLP
from hypothesis.nn.resnet import ResNetHead



class ResNet(torch.nn.Module):

    def __init__(self,
        shape_xs,
        shape_ys,
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
        trunk_dropout=None,
        trunk_layers=hypothesis.default.trunk,
        transform_output="normalize"):
        super(ResNet, self).__init__()
        # Compute the dimensionality of the inputs.
        self.dimensionality = len(shape_xs)
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
        if trunk_dropout is None:
            trunk_dropout = dropout
        # Construct the trunk of the network.
        self.trunk = MLP(
            shape_xs=(embedding_dim,),
            shape_ys=shape_ys,
            activation=trunk_activation,
            dropout=trunk_dropout,
            layers=trunk_layers,
            transform_output=transform_output)

    def forward(self, x):
        z = self.head(x)
        y = self.trunk(z)

        return y
