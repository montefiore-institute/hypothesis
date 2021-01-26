r"""Definition of the DenseNet architecture.

"""

import hypothesis as h
import torch

from .default import batchnorm as default_batchnorm
from .default import bottleneck_factor as default_bottleneck_factor
from .default import channels as default_channels
from .default import convolution_bias as default_convolution_bias
from .default import depth as default_depth
from hypothesis.nn.model.mlp import MLP
from hypothesis.nn.model.densenet import DenseNetHead


class DenseNet(torch.nn.Module):

    def __init__(self,
                 shape_xs,
                 shape_ys,
                 activation=h.default.activation,
                 batchnorm=default_batchnorm,
                 bottleneck_factor=default_bottleneck_factor,
                 channels=default_channels,
                 convolution_bias=default_convolution_bias,
                 depth=default_depth,
                 dropout=h.default.dropout,
                 trunk_activation=None,
                 trunk_dropout=h.default.dropout,
                 trunk_layers=h.default.trunk,
                 transform_output="normalize"):
        super(DenseNet, self).__init__()
        # Compute the dimensionality of the inputs.
        self._dimensionality = len(shape_xs)
        # Construct the convolutional DenseNet head.
        self._head = DenseNetHead(
            activation=activation,
            batchnorm=batchnorm,
            bottleneck_factor=bottleneck_factor,
            channels=channels,
            convolution_bias=convolution_bias,
            depth=depth,
            dropout=dropout,
            shape_xs=shape_xs)
        # Compute the embedding dimensionality of the head.
        embedding_dim = self._head.embedding_dimensionality()
        # Check if custom trunk settings have been defined.
        if trunk_activation is None:
            trunk_activation = activation
        if trunk_dropout is None:
            trunk_dropout = dropout
        # Construct the trunk of the network.
        self._trunk = MLP(
            shape_xs=(embedding_dim,),
            shape_ys=shape_ys,
            activation=trunk_activation,
            dropout=trunk_dropout,
            layers=trunk_layers,
            transform_output=transform_output)

    def forward(self, x):
        z = self._head(x)
        y = self._trunk(z)

        return y
