r"""Definition of the DenseNet head.

"""

import hypothesis as h
import numpy as np
import torch

from .default import batchnorm as default_batchnorm
from .default import bottleneck_factor as default_bottleneck_factor
from .default import channels as default_channels
from .default import convolution_bias as default_convolution_bias
from .default import depth as default_depth
from hypothesis.nn.model.densenet.util import load_modules
from hypothesis.nn.util import dimensionality


class DenseNetHead(torch.nn.Module):

    def __init__(self,
                 shape_xs,
                 activation=h.default.activation,
                 batchnorm=default_batchnorm,
                 bottleneck_factor=default_bottleneck_factor,
                 channels=default_channels,
                 convolution_bias=default_convolution_bias,
                 depth=default_depth,
                 dropout=h.default.dropout):
        super(DenseNetHead, self).__init__()
        # Infer the dimensionality from the input shape.
        self._dimensionality = len(shape_xs)
        # Dimensionality and architecture properties.
        growth_rate, in_planes, config, modules = self._load_configuration(depth)
        self._module_convolution = modules[0]
        self._module_batchnorm = modules[1]
        self._module_maxpool = modules[2]
        self._module_average_pooling = modules[3]
        self._module_adaptive_average_pooling = modules[4]
        self._module_activation = activation
        # Network properties
        self._batchnorm = batchnorm
        self._channels = channels
        self._convolution_bias = convolution_bias
        self._in_planes = in_planes
        self._shape_xs = shape_xs
        # Network structure
        self._network_head = self._build_head()
        self._network_body = self._build_body(config, bottleneck_factor, dropout, growth_rate)
        self._embedding_dim = self._embedding_dimensionality()

    def _build_head(self):
        mappings = []

        # Convolution
        mappings.append(self._module_convolution(
            self._channels,
            self._in_planes,
            bias=self._convolution_bias,
            kernel_size=7,
            padding=3,
            stride=2))
        # Batch normalization
        if self._batchnorm:
            mappings.append(self._module_batchnorm(self._in_planes))
        # Activation
        mappings.append(self._module_activation())
        # Max pooling
        mappings.append(self._module_maxpool(
            kernel_size=3,
            padding=1,
            stride=2))

        return torch.nn.Sequential(*mappings)

    def _build_body(self, config, bottleneck_factor, dropout, growth_rate):
        mappings = []
        num_features = self._in_planes
        for index, num_layers in enumerate(config):
            # DenseBlock
            mappings.append(DenseBlock(
                activation=self._module_activation,
                batchnorm=self._batchnorm,
                bottleneck_factor=bottleneck_factor,
                dimensionality=self._dimensionality,
                dropout=dropout,
                growth_rate=growth_rate,
                num_input_features=num_features,
                num_layers=num_layers))
            num_features += num_layers * growth_rate
            # Transition
            if index != len(config) - 1:
                mappings.append(self._build_transition(
                    input_features=num_features,
                    output_features=num_features // 2))
                num_features = num_features // 2
        # Batch normalization
        if self._batchnorm:
            mappings.append(self._module_batchnorm(num_features))
        # Activation
        mappings.append(self._module_activation())
        # Adaptive average pooling
        pooling_shape = [1 for _ in range(self._dimensionality)]
        mappings.append(self._module_adaptive_average_pooling(pooling_shape))

        return torch.nn.Sequential(*mappings)

    def _build_transition(self, input_features, output_features):
        mappings = []

        # Batch normalization
        if self._batchnorm:
            mappings.append(self._module_batchnorm(input_features))
        # Activation
        mappings.append(self._module_activation())
        # Convolution
        mappings.append(self._module_convolution(
            input_features,
            output_features,
            bias=self._convolution_bias,
            kernel_size=1,
            stride=1))
        # Average pooling
        mappings.append(self._module_average_pooling(
            kernel_size=2,
            stride=2))

        return torch.nn.Sequential(*mappings)

    def _embedding_dimensionality(self):
        shape = (1, self._channels) + self._shape_xs
        with torch.no_grad():
            x = torch.randn(shape)
            latents = self._network_body(self._network_head(x)).view(-1)
            dimensionality = len(latents)

        return dimensionality

    def embedding_dimensionality(self):
        return self._embedding_dim

    def forward(self, x):
        z = self._network_head(x)
        z = self._network_body(z)

        return z.view(-1, self._embedding_dim) # Flatten

    def _load_configuration(self, depth):
        modules = load_modules(self._dimensionality)
        configurations = {
            121: load_configuration_121,
            161: load_configuration_161,
            169: load_configuration_169,
            201: load_configuration_201}
        growth_rate, input_features, config = configurations[depth]()

        return growth_rate, input_features, config, modules



class DenseBlock(torch.nn.Module):

    def __init__(self,
                 dimensionality,
                 activation,
                 batchnorm,
                 bottleneck_factor,
                 dropout,
                 growth_rate,
                 num_input_features,
                 num_layers):
        super(DenseBlock, self).__init__()
        # Add the layers to the block.
        self._layers = torch.nn.ModuleList()
        for index in range(num_layers):
            self._layers.append(DenseLayer(
                activation=activation,
                batchnorm=batchnorm,
                bottleneck_factor=bottleneck_factor,
                dimensionality=dimensionality,
                dropout=dropout,
                growth_rate=growth_rate,
                num_input_features=num_input_features + index * growth_rate))

    def forward(self, x):
        features = [x]
        for layer in self._layers:
            features.append(layer(features))

        return torch.cat(features, dim=1)



class DenseLayer(torch.nn.Module):

    def __init__(self,
                 dimensionality,
                 activation,
                 batchnorm,
                 bottleneck_factor,
                 dropout,
                 growth_rate,
                 num_input_features):
        super(DenseLayer, self).__init__()
        # Load the modules depending on the dimensionality
        modules = load_modules(dimensionality)
        self._module_convolution = modules[0]
        self._module_batchnorm = modules[1]
        self._module_maxpool = modules[2]
        self._module_average_pooling = modules[3]
        self._module_activation = activation
        # Construct the dense layer
        self._network_mapping = self._build_mapping(
            batchnorm,
            bottleneck_factor,
            dropout,
            growth_rate,
            num_input_features)

    def _build_mapping(self, batchnorm, bottleneck_factor, dropout, growth_rate, num_input_features):
        mappings = []

        if batchnorm:
            mappings.append(self._module_batchnorm(num_input_features))
        mappings.append(self._module_activation())
        mappings.append(self._module_convolution(
            num_input_features,
            bottleneck_factor * growth_rate,
            kernel_size=1,
            stride=1,
            bias=False))
        mappings.append(self._module_batchnorm(bottleneck_factor * growth_rate))
        mappings.append(self._module_activation())
        mappings.append(self._module_convolution(
            bottleneck_factor * growth_rate,
            growth_rate,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False))
        # Dropout
        if dropout > 0:
            mappings.append(torch.nn.Dropout(p=dropout))

        return torch.nn.Sequential(*mappings)

    def forward(self, x):
        z = torch.cat(x, dim=1)

        return self._network_mapping(z)


def load_configuration_121():
    growth_rate = 32
    input_features = 64
    config = [6, 12, 24, 16]

    return growth_rate, input_features, config


def load_configuration_161():
    growth_rate = 48
    input_features = 96
    config = [6, 12, 36, 24]

    return growth_rate, input_features, config


def load_configuration_169():
    growth_rate = 32
    input_features = 64
    config = [6, 12, 32, 32]

    return growth_rate, input_features, config


def load_configuration_201():
    growth_rate = 32
    input_features = 64
    config = [6, 12, 48, 32]

    return growth_rate, input_features, config
