import hypothesis
import numpy as np
import torch

from hypothesis.nn.util import allocate_output_transform
from hypothesis.nn.util import compute_dimensionality



class DenseNet(torch.nn.Module):

    def __init__(self,
        config,
        shape_xs,
        shape_ys=(1,),
        activation=hypothesis.default.activation,
        batchnorm=True,
        bottleneck_factor=4,
        channels=3,
        convolution_bias=False,
        dense_dropout=hypothesis.default.dropout,
        growth_rate=32,
        in_planes=64,
        trunk=hypothesis.default.trunk,
        trunk_dropout=hypothesis.default.dropout,
        ys_transform=hypothesis.default.output_transform):
        super(DenseNet, self).__init__()
        # Infer the dimensionality from the input shape.
        dimensionality = len(shape_xs)
        # Dimensionality and architecture properties.
        self.module_convolution = modules[0]
        self.modules_batchnorm = modules[1]
        self.module_maxpool = modules[2]
        self.module_activation = activation
        # Network properties
        self.batchnorm = batchnorm
        self.channels = channels
        self.convolution_bias = convolution_bias
        self.in_planes = in_planes
        self.shape_xs = shape_xs
        self.shape_ys = shape_ys
        # Network structure
        self.network_head = self._build_head()
        self.network_body = self._build_body()
        self.network_trunk = self._build_trunk(trunk, trunk_dropout, ys_transform)

    def _build_head(self):
        mappings = []

        # Convolution
        mappings.append(self.module_convolution(
            self.channels,
            self.in_planes,
            bias=self.convolution_bias,
            kernel_size=7,
            padding=3,
            stride=2))
        # Batch normalization
        if self.batchnorm:
            mappings.append(self.module_batchnorm(self.in_planes))
        # Activation
        mappings.append(self.module_activation(inplace=True))
        # Max pooling
        mappings.append(self.module_maxpool(
            kernel_size=3,
            padding=1,
            stride=2))

        return torch.nn.Sequential(*mappings)

    def _build_body(self):
        raise NotImplementedError

    def _build_trunk(self, config, dropout, ys_transform):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def _load_configuration(self, dimensionality, depth):
        modules = load_modules(dimensionality)
        raise NotImplementedError



def load_modules(dimensionality):
    raise NotImplementedError
