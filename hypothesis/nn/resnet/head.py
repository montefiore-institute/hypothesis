import hypothesis
import hypothesis.nn
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
from hypothesis.nn.resnet.util import load_modules
from hypothesis.nn.util import compute_dimensionality




class ResNetHead(torch.nn.Module):

    def __init__(self,
        shape_xs,
        activation=hypothesis.default.activation,
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
        self.dimensionality = len(shape_xs)
        # Dimensionality and architecture properties.
        self.block, self.blocks_per_layer, modules = self._load_configuration(depth)
        self.module_convolution = modules[0]
        self.module_batchnorm = modules[1]
        self.module_maxpool = modules[2]
        self.module_adaptive_avg_pool = modules[3]
        self.module_activation = activation
        # Network properties.
        self.batchnorm = batchnorm
        self.channels = channels
        self.convolution_bias = convolution_bias
        self.dilate = dilate
        self.dilation = 1
        self.groups = groups
        self.in_planes = in_planes
        self.shape_xs = shape_xs
        self.shape_ys = shape_ys
        self.width_per_group = width_per_group
        # Network structure
        self.network_head = self._build_head()
        self.network_body = self._build_body()
        self.embedding_dim = self._embedding_dimensionality()

    def _build_head(self):
        mappings = []
        # Convolution
        mappings.append(self.module_convolution(
            self.channels,
            self.in_planes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=self.convolution_bias))
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
        mappings = []
        exponent = int(np.log2(self.in_planes))
        stride = 1
        for layer_index, blocks in enumerate(self.blocks_per_layer):
            planes = 2 ** (exponent + layer_index)
            mappings.append(self._build_layer(
                planes=planes,
                blocks=blocks,
                stride=stride))
            stride = 2
        # Adaptive average pooling.
        shape = [1 for _ in range(self.dimensionality)]
        mappings.append(self.module_adaptive_avg_pool(shape))

        return torch.nn.Sequential(*mappings)

    def _build_layer(self, planes, blocks, stride):
        mappings = []
        previous_dilation = self.dilation
        if self.dilate:
            self.dilation *= stride
            stride = 1
        # Check if a downsampling function needs to be allocated.
        if stride != 1 or self.in_planes != planes * self.block.EXPANSION:
                downsample = torch.nn.Sequential(self.module_convolution(
                    self.in_planes,
                    planes * self.block.EXPANSION,
                    bias=self.convolution_bias,
                    kernel_size=1,
                    stride=stride),
                self.module_batchnorm(planes * self.block.EXPANSION))
        else:
            downsample = None
        # Allocate the blocks in the current layer.
        mappings.append(self.block(
            activation=self.module_activation,
            dimensionality=self.dimensionality,
            downsample=downsample,
            groups=self.groups,
            in_planes=self.in_planes,
            out_planes=planes,
            stride=stride,
            width_per_group=self.width_per_group))
        self.in_planes = planes * self.block.EXPANSION
        for _ in range(1, blocks):
            mappings.append(self.block(
                activation=self.module_activation,
                batchnorm=self.batchnorm,
                dilation=self.dilation,
                dimensionality=self.dimensionality,
                groups=self.groups,
                in_planes=self.in_planes,
                out_planes=planes,
                width_per_group=self.width_per_group))

        return torch.nn.Sequential(*mappings)

    def _load_configuration(self, depth):
        modules = load_modules(self.dimensionality)
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
        shape = (1, self.channels) + self.shape_xs
        with torch.no_grad():
            x = torch.randn(shape)
            latents = self.network_body(self.network_head(x)).view(-1)
            dimensionality = len(latents)

        return dimensionality

    def embedding_dimensionality(self):
        return self.embedding_dim

    def forward(self, x):
        z = self.network_head(x)
        z = self.network_body(z)

        return z



class BasicBlock(torch.nn.Module):
    EXPANSION = 1

    def __init__(self, dimensionality,
        in_planes,
        out_planes,
        activation=hypothesis.default.activation,
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
        self.module_convolution = modules[0]
        self.module_batchnorm = modules[1]
        self.module_maxpool = modules[2]
        self.module_adaptive_avg_pool = modules[3]
        # Block properties.
        self.module_activation = activation
        self.activation = activation(inplace=True)
        self.bias = bias
        self.batchnorm = batchnorm
        self.dilation = dilation
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride
        # Assign the downsampling mapping, if specified.
        self.downsample_mapping = downsample
        # Build the residual mapping.
        self.residual_mapping = self._build_residual_mapping()

    def _build_residual_mapping(self):
        mappings = []

        # Convolution
        mappings.append(self.module_convolution(
            self.in_planes,
            self.out_planes,
            bias=self.bias,
            dilation=self.dilation,
            kernel_size=3,
            padding=self.dilation,
            stride=self.stride))
        # Batch normalization
        if self.batchnorm:
            mappings.append(self.module_batchnorm(self.out_planes))
            # Activation
        mappings.append(self.module_activation(inplace=True))
        # Convolution
        mappings.append(self.module_convolution(
            self.out_planes,
            self.out_planes,
            bias=self.bias,
            dilation=self.dilation,
            kernel_size=3,
            padding=self.dilation,
            stride=1))
        # Batch normalization
        if self.batchnorm:
            mappings.append(self.module_batchnorm(self.out_planes))

        return torch.nn.Sequential(*mappings)

    def forward(self, x):
        identity = x
        if self.downsample_mapping is not None:
            identity = self.downsample_mapping(identity)
        y = self.activation(identity + self.residual_mapping(x))

        return y



class Bottleneck(torch.nn.Module):
    EXPANSION = 4

    def __init__(self, dimensionality,
        in_planes,
        out_planes,
        activation=hypothesis.default.activation,
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
        self.module_convolution = modules[0]
        self.module_batchnorm = modules[1]
        self.module_maxpool = modules[2]
        self.module_adaptive_avg_pool = modules[3]
        # Block properties.
        self.module_activation = activation
        self.activation = activation(inplace=True)
        self.bias = bias
        self.batchnorm = batchnorm
        self.dilation = dilation
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride
        self.groups = groups
        self.width_per_group = width_per_group
        self.width = int(self.out_planes * (self.width_per_group // 64)) * self.groups
        # Assign the downsampling mapping, if specified.
        self.downsample_mapping = downsample
        # Build the residual mapping.
        self.residual_mapping = self._build_residual_mapping()

    def _build_residual_mapping(self):
        mappings = []

        # Convolution
        mappings.append(self.module_convolution(
            self.in_planes,
            self.width,
            bias=self.bias,
            kernel_size=1,
            stride=1))
        # Batch normalization
        if self.batchnorm:
            mappings.append(self.module_batchnorm(self.width))
        # Activation
        mappings.append(self.module_activation(inplace=True))
        # Convolution
        mappings.append(self.module_convolution(
            self.width,
            self.width,
            kernel_size=3,
            stride=self.stride,
            groups=self.groups,
            dilation=self.dilation,
            padding=self.dilation,
            bias=self.bias))
        # Batch normalization
        if self.batchnorm:
            mappings.append(self.module_batchnorm(self.width))
        # Activation
        mappings.append(self.module_activation(inplace=True))
        # Convolution
        mappings.append(self.module_convolution(
            self.width,
            self.out_planes * self.EXPANSION,
            bias=self.bias,
            kernel_size=1,
            stride=1))
        # Batch normalization
        if self.batchnorm:
            mappings.append(self.module_batchnorm(self.out_planes * self.EXPANSION))

        return torch.nn.Sequential(*mappings)

    def forward(self, x):
        identity = x
        if self.downsample_mapping is not None:
            identity = self.downsample_mapping(identity)
        y = self.activation(identity + self.residual_mapping(x))

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
