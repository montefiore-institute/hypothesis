import hypothesis
import hypothesis.nn.densenet
import torch

from .default import batchnorm as default_batchnorm
from .default import bottleneck_factor as default_bottleneck_factor
from .default import channels as default_channels
from .default import convolution_bias as default_convolution_bias
from .default import depth as default_depth
from hypothesis.nn import MLP
from hypothesis.nn.densenet import DenseNetHead



class DenseNet(torch.nn.Module):

    def __init__(self,
        shape_xs,
        shape_ys,
        activation=hypothesis.default.activation,
        batchnorm=default_batchnorm,
        bottleneck_factor=default_bottleneck_factor,
        channels=default_channels,
        convolution_bias=default_convolution_bias,
        depth=default_depth,
        dropout=hypothesis.default.dropout,
        trunk_activation=None,
        trunk_dropout=None,
        trunk_layers=hypothesis.default.trunk,
        transform_output="normalize"):
        super(DenseNet, self).__init__()
        # Compute the dimensionality of the inputs.
        self.dimensionality = len(shape_xs)
        # Construct the convolutional DenseNet head.
        self.head = DenseNetHead(
            activation=activation,
            batchnorm=batchnorm,
            bottleneck_factor=bottleneck_factor,
            channels=channels,
            convolution_bias=convolution_bias,
            depth=depth,
            dropout=dropout,
            shape_xs=shape_xs)
        # Compute the embedding dimensionality of the head.
        embedding_dim = self.head.embedding_dimensionality()
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
