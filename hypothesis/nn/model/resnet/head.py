r"""Definition of the ResNet header.

"""

import hypothesis as h
import numpy as np
import torch

from .default import batchnorm as default_batchnorm
from .default import channels as default_channels
from .default import convolution_bias as default_convolution_bias
from .default import depth as default_depth
from .default import dilate as default_dilate
from .default import groups as default_groups
from .default import in_planes as default_in_planes
from .default import width_per_group as default_width_per_group
from hypothesis.nn.model.resnet.util import load_modules
from hypothesis.nn.util import dimensionality


class ResNetHead(torch.nn.Module):

    def __init__(self,
        shape_xs,
        activation=h.default.activation,
        batchnorm=default_batchnorm,
        channels=default_channels,
        convolution_bias=default_convolution_bias,
        depth=default_depth,
        dilate=default_dilate,
        groups=default_groups,
        in_planes=default_in_planes,
        width_per_group=default_width_per_group):
        super(ResNetHead, self).__init__()
        # Infer the dimensionality from the input shape.
        self._dimensionality = len(shape_xs)
        # Dimensionality and architecture properties.
        self._block, self._blocks_per_layer, modules = self._load_configuration(int(depth))
        self._module_convolution = modules[0]
        self._module_batchnorm = modules[1]
        self._module_maxpool = modules[2]
        self._module_adaptive_avg_pool = modules[3]
        self._module_activation = activation
        # Network properties.
        self._batchnorm = batchnorm
        self._channels = channels
        self._convolution_bias = convolution_bias
        self._dilate = dilate
        self._dilation = 1
        self._groups = groups
        self._in_planes = in_planes
        self._shape_xs = shape_xs
        self._width_per_group = width_per_group
        # Network structure
        self._network_head = self._build_head()
        self._network_body = self._build_body()
        self._embedding_dim = self._embedding_dimensionality()

    def _build_head(self):
        mappings = []
        # Convolution
        mappings.append(self._module_convolution(
            self._channels,
            self._in_planes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=self._convolution_bias))
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

    def _build_body(self):
        mappings = []
        exponent = int(np.log2(self._in_planes))
        stride = 1
        for layer_index, blocks in enumerate(self._blocks_per_layer):
            planes = 2 ** (exponent + layer_index)
            mappings.append(self._build_layer(
                planes=planes,
                blocks=blocks,
                stride=stride))
            stride = 2
        # Adaptive average pooling.
        shape = [1 for _ in range(self._dimensionality)]
        mappings.append(self._module_adaptive_avg_pool(shape))

        return torch.nn.Sequential(*mappings)

    def _build_layer(self, planes, blocks, stride):
        mappings = []
        previous_dilation = self._dilation
        if self._dilate:
            self._dilation *= stride
            stride = 1
        # Check if a downsampling function needs to be allocated.
        if stride != 1 or self._in_planes != planes * self._block.EXPANSION:
                downsample = torch.nn.Sequential(self._module_convolution(
                    self._in_planes,
                    planes * self._block.EXPANSION,
                    bias=self._convolution_bias,
                    kernel_size=1,
                    stride=stride),
                self._module_batchnorm(planes * self._block.EXPANSION))
        else:
            downsample = None
        # Allocate the blocks in the current layer.
        mappings.append(self._block(
            activation=self._module_activation,
            dimensionality=self._dimensionality,
            downsample=downsample,
            groups=self._groups,
            in_planes=self._in_planes,
            out_planes=planes,
            stride=stride,
            width_per_group=self._width_per_group))
        self._in_planes = planes * self._block.EXPANSION
        for _ in range(1, blocks):
            mappings.append(self._block(
                activation=self._module_activation,
                batchnorm=self._batchnorm,
                dilation=self._dilation,
                dimensionality=self._dimensionality,
                groups=self._groups,
                in_planes=self._in_planes,
                out_planes=planes,
                width_per_group=self._width_per_group))

        return torch.nn.Sequential(*mappings)

    def _load_configuration(self, depth):
        modules = load_modules(self._dimensionality)
        configurations = {
            18: load_configuration_18,
            34: load_configuration_34,
            50: load_configuration_50,
            101: load_configuration_101,
            152: load_configuration_152}
        # Check if the desired configuration exists.
        if depth not in configurations.keys():
            raise ValueError("The specified ResNet configuration (", depth, ") does not exist.")
        configuration_loader = configurations[depth]
        block, blocks_per_layer = configuration_loader()

        return block, blocks_per_layer, modules

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

        return z.squeeze()


class BasicBlock(torch.nn.Module):
    EXPANSION = 1

    def __init__(self, dimensionality,
        in_planes,
        out_planes,
        activation=h.default.activation,
        batchnorm=True,
        bias=False,
        dilation=1,
        downsample=None,
        stride=1,
        groups=1,
        width_per_group=64):
        super(BasicBlock, self).__init__()
        # Load the requested modules depending on the dimensionality.
        modules = load_modules(dimensionality)
        self._module_convolution = modules[0]
        self._module_batchnorm = modules[1]
        self._module_maxpool = modules[2]
        self._module_adaptive_avg_pool = modules[3]
        # Block properties.
        self._module_activation = activation
        self._activation = activation()
        self._bias = bias
        self._batchnorm = batchnorm
        self._dilation = dilation
        self._in_planes = in_planes
        self._out_planes = out_planes
        self._stride = stride
        # Assign the downsampling mapping, if specified.
        self._downsample_mapping = downsample
        # Build the residual mapping.
        self._residual_mapping = self._build_residual_mapping()

    def _build_residual_mapping(self):
        mappings = []

        # Convolution
        mappings.append(self._module_convolution(
            self._in_planes,
            self._out_planes,
            bias=self._bias,
            dilation=self._dilation,
            kernel_size=3,
            padding=self._dilation,
            stride=self._stride))
        # Batch normalization
        if self._batchnorm:
            mappings.append(self._module_batchnorm(self._out_planes))
            # Activation
        mappings.append(self._module_activation())
        # Convolution
        mappings.append(self._module_convolution(
            self._out_planes,
            self._out_planes,
            bias=self._bias,
            dilation=self._dilation,
            kernel_size=3,
            padding=self._dilation,
            stride=1))
        # Batch normalization
        if self._batchnorm:
            mappings.append(self._module_batchnorm(self._out_planes))

        return torch.nn.Sequential(*mappings)

    def forward(self, x):
        identity = x
        if self._downsample_mapping is not None:
            identity = self._downsample_mapping(identity)
        y = self._activation(identity + self._residual_mapping(x))

        return y



class Bottleneck(torch.nn.Module):
    EXPANSION = 4

    def __init__(self, dimensionality,
        in_planes,
        out_planes,
        activation=h.default.activation,
        batchnorm=True,
        bias=False,
        dilation=1,
        downsample=None,
        stride=1,
        groups=1,
        width_per_group=64):
        super(Bottleneck, self).__init__()
        # Load the requested modules depending on the dimensionality.
        modules = load_modules(dimensionality)
        self._module_convolution = modules[0]
        self._module_batchnorm = modules[1]
        self._module_maxpool = modules[2]
        self._module_adaptive_avg_pool = modules[3]
        # Block properties.
        self._module_activation = activation
        self._activation = activation()
        self._bias = bias
        self._batchnorm = batchnorm
        self._dilation = dilation
        self._in_planes = in_planes
        self._out_planes = out_planes
        self._stride = stride
        self._groups = groups
        self._width_per_group = width_per_group
        self._width = int(self._out_planes * (self._width_per_group // 64)) * self._groups
        # Assign the downsampling mapping, if specified.
        self._downsample_mapping = downsample
        # Build the residual mapping.
        self._residual_mapping = self._build_residual_mapping()

    def _build_residual_mapping(self):
        mappings = []

        # Convolution
        mappings.append(self._module_convolution(
            self._in_planes,
            self._width,
            bias=self._bias,
            kernel_size=1,
            stride=1))
        # Batch normalization
        if self._batchnorm:
            mappings.append(self._module_batchnorm(self._width))
        # Activation
        mappings.append(self._module_activation())
        # Convolution
        mappings.append(self._module_convolution(
            self._width,
            self._width,
            kernel_size=3,
            stride=self._stride,
            groups=self._groups,
            dilation=self._dilation,
            padding=self._dilation,
            bias=self._bias))
        # Batch normalization
        if self._batchnorm:
            mappings.append(self._module_batchnorm(self._width))
        # Activation
        mappings.append(self._module_activation())
        # Convolution
        mappings.append(self._module_convolution(
            self._width,
            self._out_planes * self._EXPANSION,
            bias=self._bias,
            kernel_size=1,
            stride=1))
        # Batch normalization
        if self._batchnorm:
            mappings.append(self._module_batchnorm(self._out_planes * self._EXPANSION))

        return torch.nn.Sequential(*mappings)

    def forward(self, x):
        identity = x
        if self._downsample_mapping is not None:
            identity = self._downsample_mapping(identity)
        y = self._activation(identity + self._residual_mapping(x))

        return y



def load_configuration_18():
    return BasicBlock, [2, 2, 2, 2]


def load_configuration_34():
    return BasicBlock, [3, 4, 6, 3]


def load_configuration_50():
    return Bottleneck, [3, 4, 6, 3]


def load_configuration_101():
    return Bottleneck, [3, 4, 23, 3]


def load_configuration_152():
    return Bottleneck, [3, 8, 36, 3]
