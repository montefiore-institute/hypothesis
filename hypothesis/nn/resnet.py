import numpy as np
import torch



class ResNet(torch.nn.Module):

    def __init__(self, depth,
                 activation=None,
                 batchnorm=True,
                 channels=3,
                 convolution_bias=True,
                 dilate=False,
                 trunk=[4096, 4096, 4096, 1],
                 trunk_dropout=0.0):
        super(ResNet, self).__init__()
        # Load the specified ResNet configuration.
        self.block, self.layer_blocks = self._load_configuration(depth)
        # Check if a custom activation function has been specified.
        if activation is None:
            activation = torch.nn.ReLU
        # Network properties
        self.activation = activation
        self.batchnorm = batchnorm
        self.channels = channels
        self.convolution_bias = convolution_bias
        self.dilate = dilate
        self.dilation = 1
        self.final_planes = 0
        self.in_planes = 64
        self.trunk = trunk
        self.trunk_dropout = trunk_dropout
        # Build the model structure according to the configuration.
        self.network_head = self._build_head()
        self.network_body = self._build_body()
        self.network_trunk = self._build_trunk()

    def _build_head(self):
        head = []
        convolution = torch.nn.Conv2d(
            self.channels,
            self.in_planes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=self.convolution_bias)
        activation = self.activation(inplace=True)
        max_pooling = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        head.append(convolution)
        if self.batchnorm:
            batch_normalization = torch.nn.BatchNorm2d(self.in_planes)
            head.append(batch_normalization)
        head.append(activation)
        head.append(max_pooling)
        head = torch.nn.Sequential(*head)

        return head

    def _build_body(self):
        layers = []
        stride = 1
        exponent = int(np.log2(self.in_planes))
        for layer_index, blocks in enumerate(self.layer_blocks):
            planes = 2 ** (exponent + layer_index)
            layer = self._build_layer(planes=planes, blocks=blocks, stride=stride)
            stride = 2
            layers.append(layer)
        layers.append(torch.nn.AdaptiveAvgPool2d((1, 1)))
        body = torch.nn.Sequential(*layers)
        self.final_planes = planes

        return body

    def _build_layer(self, planes, blocks, stride):
        previous_dilation = self.dilation
        # Check if a dilated convolution is required.
        if self.dilate:
            self.dilation *= stride
            stride = 1
        new_dimensionality = planes * self.block.expansion
        if stride != 1 or self.in_planes != new_dimensionality:
            conv = torch.nn.Conv2d(self.in_planes, new_dimensionality, kernel_size=1, stride=stride, bias=False)
            if self.batchnorm:
                downsample = torch.nn.Sequential(conv, torch.nn.BatchNorm2d(new_dimensionality))
            else:
                downsample = conv
        else:
            downsample = None
        # Build the sequence of blocks.
        layers = []
        block = self.block(self.in_planes, planes, stride, self.activation, previous_dilation, downsample, self.batchnorm)
        layers.append(block)
        self.in_planes = planes * self.block.expansion
        stride = 1
        for _ in range(1, blocks):
            block = self.block(self.in_planes, planes, stride, self.activation, self.dilation, downsample=None, batchnorm=self.batchnorm)
            layers.append(block)

        return torch.nn.Sequential(*layers)

    def _build_trunk(self):
        layers = []
        dimensionality = self.final_planes * self.block.expansion
        layers.append(torch.nn.Linear(dimensionality, self.trunk[0]))
        for i in range(1, len(self.trunk)):
            layers.append(self.activation())
            layers.append(torch.nn.Linear(self.trunk[i - 1], self.trunk[i]))
            # Check if dropout needs to be added.
            if self.trunk_dropout > 0:
                layers.append(torch.nn.Dropout(p=self.trunk_dropout))

        return torch.nn.Sequential(*layers)

    def forward(self, xs):
        latents = self.network_head(xs)
        latents = self.network_body(latents)
        latents = latents.reshape(latents.size(0), -1) # Flatten
        log_ratio = self.network_trunk(latents)

        return log_ratio

    def _load_configuration(self, depth):
        # Build the configuration mapping.
        mapping = {
            18: self._load_configuration_18,
            34: self._load_configuration_34,
            50: self._load_configuration_50,
            101: self._load_configuration_101,
            152: self._load_configuration_152}
        # Check if the desired configuration exists.
        if depth not in mapping.keys():
            raise ValueError("The specified ResNet configuration (", depth, ") does not exist.")
        loader = mapping[depth]

        return loader()

    @staticmethod
    def _load_configuration_18():
        return BasicBlock, [2, 2, 2, 2]

    @staticmethod
    def _load_configuration_34():
        return BasicBlock, [3, 4, 6, 3]

    @staticmethod
    def _load_configuration_50():
        return Bottleneck, [3, 4, 6, 3]

    @staticmethod
    def _load_configuration_101():
        return Bottleneck, [3, 4, 23, 3]

    @staticmethod
    def _load_configuration_152():
        return Bottleneck, [3, 8, 36, 3]



class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, activation=None, dilation=1, downsample=None, batchnorm=True):
        super(BasicBlock, self).__init__()
        # Build the forward model.
        layers = []
        self.downsample = downsample
        self.activation_function = activation(inplace=True)
        layer = torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        layers.append(layer)
        if batchnorm:
            layers.append(torch.nn.BatchNorm2d(out_planes))
        layers.append(activation(inplace=True))
        layer = torch.nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        layers.append(layer)
        if batchnorm:
            layers.append(torch.nn.BatchNorm2d(out_planes))
        self.model = torch.nn.Sequential(*layers)
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(identity)
        y = identity + self.model(x)

        return self.activation_function(y)



class Bottleneck(torch.nn.Module):
    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1, activation=None, dilation=1, downsample=None, batchnorm=True):
        super(Bottleneck, self).__init__()
        # Build the forward function.
        layers = []
        self.downsample = downsample
        self.activation_function = activation(inplace=True)
        layers.append(torch.nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1))
        if batchnorm:
            layers.append(torch.nn.BatchNorm2d(out_planes))
        layers.append(activation(inplace=True))
        layers.append(torch.nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation))
        if batchnorm:
            layers.append(torch.nn.BatchNorm2d(out_planes))
        layers.append(activation(inplace=True))
        layers.append(torch.nn.Conv2d(out_planes, out_planes * self.expansion, kernel_size=1, stride=1))
        if batchnorm:
            layers.append(torch.nn.BatchNorm2d(out_planes * self.expansion))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(identity)
        y = identity + self.model(x)

        return self.activation_function(y)



class ResNet18(ResNet):

    def __init__(self, activation=None,
                 batchnorm=True,
                 channels=3,
                 convolution_bias=True,
                 dilate=False,
                 trunk=[4096, 4096, 4096, 1],
                 trunk_dropout=0.0):
        depth = 18
        super(ResNet18, self).__init__(
            depth=depth,
            activation=activation,
            batchnorm=batchnorm,
            channels=channels,
            convolution_bias=convolution_bias,
            dilate=dilate,
            trunk=trunk,
            trunk_dropout=trunk_dropout)



class ResNet34(ResNet):

    def __init__(self, activation=None,
                 batchnorm=True,
                 channels=3,
                 convolution_bias=True,
                 dilate=False,
                 trunk=[4096, 4096, 4096, 1],
                 trunk_dropout=0.0):
        depth = 34
        super(ResNet34, self).__init__(
            depth=depth,
            activation=activation,
            batchnorm=batchnorm,
            channels=channels,
            convolution_bias=convolution_bias,
            dilate=dilate,
            trunk=trunk,
            trunk_dropout=trunk_dropout)



class ResNet50(ResNet):

    def __init__(self, activation=None,
                 batchnorm=True,
                 channels=3,
                 convolution_bias=True,
                 dilate=False,
                 trunk=[4096, 4096, 4096, 1],
                 trunk_dropout=0.0):
        depth = 50
        super(ResNet50, self).__init__(
            depth=depth,
            activation=activation,
            batchnorm=batchnorm,
            channels=channels,
            convolution_bias=convolution_bias,
            dilate=dilate,
            trunk=trunk,
            trunk_dropout=trunk_dropout)



class ResNet101(ResNet):

    def __init__(self, activation=None,
                 batchnorm=True,
                 channels=3,
                 convolution_bias=True,
                 dilate=False,
                 trunk=[4096, 4096, 4096, 1],
                 trunk_dropout=0.0):
        depth = 101
        super(ResNet101, self).__init__(
            depth=depth,
            activation=activation,
            batchnorm=batchnorm,
            channels=channels,
            convolution_bias=convolution_bias,
            dilate=dilate,
            trunk=trunk,
            trunk_dropout=trunk_dropout)



class ResNet152(ResNet):

    def __init__(self, activation=None,
                 batchnorm=True,
                 channels=3,
                 convolution_bias=True,
                 dilate=False,
                 trunk=[4096, 4096, 4096, 1],
                 trunk_dropout=0.0):
        depth = 152
        super(ResNet152, self).__init__(
            depth=depth,
            activation=activation,
            batchnorm=batchnorm,
            channels=channels,
            convolution_bias=convolution_bias,
            dilate=dilate,
            trunk=trunk,
            trunk_dropout=trunk_dropout)
