import hypothesis
import numpy as np
import torch

from hypothesis.nn.util import allocate_output_transform
from hypothesis.nn.util import compute_dimensionality



class DenseNet(torch.nn.Module):

    def __init__(self,
        depth,
        shape_xs,
        shape_ys=(1,),
        activation=hypothesis.default.activation,
        batchnorm=True,
        bottleneck_factor=4,
        channels=3,
        convolution_bias=False,
        dense_dropout=hypothesis.default.dropout,
        trunk=hypothesis.default.trunk,
        trunk_dropout=hypothesis.default.dropout,
        ys_transform=hypothesis.default.output_transform):
        super(DenseNet, self).__init__()
        # Infer the dimensionality from the input shape.
        self.dimensionality = len(shape_xs)
        # Dimensionality and architecture properties.
        growth_rate, in_planes, config, modules = self._load_configuration(depth)
        self.module_convolution = modules[0]
        self.modules_batchnorm = modules[1]
        self.module_maxpool = modules[2]
        self.module_average_pooling = modules[3]
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
        self.network_body = self._build_body(config, bottleneck_factor, dense_dropout, growth_rate)
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

    def _build_body(self, config, bottleneck_factor, dropout, growth_rate):
        mappings = []
        num_features = self.in_planes
        for index, num_layers in enumerate(config):
            # DenseBlock
            mappings.append(DenseBlock(
                activation=self.module_activation,
                batchnorm=self.batchnorm,
                bottleneck_factor=bottleneck_factor,
                dimensionality=self.dimensionality,
                dropout=dropout,
                growth_rate=self.growth_rate,
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
        if self.batchnorm:
            mappings.append(self.module_batchnorm(num_features))

        return torch.nn.Sequential(*mappings)

    def _build_transition(self, input_features, output_features):
        mappings = []

        # Batch normalization
        if self.batchnorm:
            mappings.append(self.module_batchnorm(input_features))
        # Activation
        mappings.append(self.module_activation(inplace=True))
        # Convolution
        mappings.append(self.module_convolution(
            input_features,
            output_features,
            bias=self.convolution_bias,
            kernel_size=1,
            stride=1))
        # Average pooling
        mappings.append(self.module_average_pooling(
            kernel_size=2,
            stride=2))

        return torch.nn.Sequential(*mappings)

    def _build_trunk(self, config, dropout, ys_transform):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def _load_configuration(self, depth):
        modules = load_modules(self.dimensionality)
        configurations = {
            121: _load_configuration_121,
            161: _load_configuration_161,
            169: _load_configuration_169,
            201: _load_configuration_201}
        growth_rate, input_features, layers = configurations[depth]()

        return growth_rate, input_features, config, modules

    @staticmethod
    def _load_configuration_121(dimensionality):
        growth_rate = 32
        input_features = 64
        config = [6, 12, 24, 16]

        return growth_rate, input_features, config

    @staticmethod
    def _load_configuration_161(dimensionality):
        growth_rate = 48
        input_features = 96
        config = [6, 12, 36, 24]

        return growth_rate, input_features, config

    @staticmethod
    def _load_configuration_169(dimensionality):
        growth_rate = 32
        input_features = 64
        config = [6, 12, 32, 32]

        return growth_rate, input_features, config

    @staticmethod
    def _load_configuration_201(dimensionality):
        growth_rate = 32
        input_features = 64
        config = [6, 12, 48, 32]

        return growth_rate, input_features, config



class DenseBlock(torch.nn.Module):

    def __init__(self, dimensionality,
        activation,
        batchnorm,
        bottleneck_factor,
        dropout,
        growth_rate,
        num_input_features,
        num_layers):
        super(DenseBlock, self).__init__()
        # Add the layers to the block.
        self.layers = []
        for index in range(num_layers):
            self.layers.append(DenseLayer(
                activation=activation,
                batchnorm=batchnorm,
                bottleneck_factor=bottleneck_factor,
                dimensionality=dimensionality,
                dropout=dropout,
                num_input_features=num_input_features + index * growth_rate))

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            features.append(layer(features))

        return torch.cat(features, dim=1)



class DenseLayer(torch.nn.Module):

    def __init__(self, dimensionality,
        activation,
        batchnorm,
        bottleneck_factor_factor,
        dropout,
        num_input_features):
        super(DenseLayer, self).__init__()
        # Load the modules depending on the dimensionality
        modules = load_modules(dimensionality)
        self.module_convolution = modules[0]
        self.modules_batchnorm = modules[1]
        self.module_maxpool = modules[2]
        self.module_average_pooling = modules[3]
        self.module_activation = activation
        # Construct the dense layer
        self.network_mapping = self._build_mapping(batchnorm,
            bottleneck_factor,
            dropout,
            num_input_features)

    def _build_mapping(self, batchnorm, bottleneck_factor, dropout, num_input_features):
        mappings = []

        # Bottleneck
        # Batch normalization
        if batchnorm:
            mappings.append(self.module_batchnorm(num_input_features))
        # Activation
        mappings.append(self.module_activation(inplace=True))
        # Convolution
        mappings.append(self.module_convolution(
            num_input_features,
            bottleneck_factor * growth_rate,
            kernel_size=1,
            stride=1,
            bias=False))
        # Normalization
        mappings.append(self.module_batchnorm(bottleneck_factor * growth_rate))
        # Activation
        mappings.append(self.module_activation(inplace=True))
        # Convolution
        mappings.append(self.module_convolution(
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
        return self.network_mapping(x)



def load_modules(dimensionality):
    configurations = {
        1: load_modules_1_dimensional,
        2: load_modules_2_dimensional,
        3: load_modules_3_dimensional}

    return configurations[dimensionality]()


def load_modules_1_dimensional():
    c = torch.nn.Conv1d
    b = torch.nn.BatchNorm1d
    m = torch.nn.MaxPool1d
    a = torch.nn.AvgPool1d

    return c, b, m, a


def load_modules_2_dimensional():
    c = torch.nn.Conv2d
    b = torch.nn.BatchNorm2d
    m = torch.nn.MaxPool2d
    a = torch.nn.AvgPool2d

    return c, b, m, a


def load_modules_3_dimensional():
    c = torch.nn.Conv3d
    b = torch.nn.BatchNorm3d
    m = torch.nn.MaxPool3d
    a = torch.nn.AvgPool3d

    return c, b, m, a
