import torch

from hypothesis.nn import BaseRatioEstimator
from hypothesis.nn import ResNet



class ResNetRatioEstimator(BaseRatioEstimator):

    def __init__(self, depth,
                 activation=None,
                 batchnorm=True,
                 channels=3,
                 convolution_bias=True,
                 dilate=False,
                 trunk=[4096, 4096, 4096],
                 trunk_dropout=0.0):
        super(BaseRatioEstimator, self).__init__()
        trunk.append(1) # Add final output.
        self.resnet = ResNet(depth,
            activation=activation,
            batchnorm=batchnorm,
            channels=channels,
            convolution_bias=convolution_bias,
            dilate=dilate,
            trunk=trunk,
            trunk_dropout=trunk_dropout)

    def forward(self, xs):
        log_ratios = self.resnet(xs)

        return log_ratios.sigmoid(), log_ratios

    def log_ratio(self, xs):
        log_ratios = self.resnet(xs)

        return log_ratios



class ResNet18RatioEstimator(ResNetRatioEstimator):

    def __init__(self, activation=None,
                 batchnorm=True,
                 channels=3,
                 convolution_bias=True,
                 dilate=False,
                 trunk=[4096, 4096, 4096],
                 trunk_dropout=0.0):
        depth = 18
        super(ResNet18RatioEstimator, self).__init__(
            activation=activation,
            batchnorm=batchnorm,
            channels=channels,
            convolution_bias=convolution_bias,
            dilate=dilate,
            trunk=trunk,
            trunk_dropout=trunk_dropout)



class ResNet34RatioEstimator(ResNetRatioEstimator):

    def __init__(self, activation=None,
                 batchnorm=True,
                 channels=3,
                 convolution_bias=True,
                 dilate=False,
                 trunk=[4096, 4096, 4096],
                 trunk_dropout=0.0):
        depth = 34
        super(ResNet34RatioEstimator, self).__init__(
            activation=activation,
            batchnorm=batchnorm,
            channels=channels,
            convolution_bias=convolution_bias,
            dilate=dilate,
            trunk=trunk,
            trunk_dropout=trunk_dropout)



class ResNet50RatioEstimator(ResNetRatioEstimator):

    def __init__(self, activation=None,
                 batchnorm=True,
                 channels=3,
                 convolution_bias=True,
                 dilate=False,
                 trunk=[4096, 4096, 4096],
                 trunk_dropout=0.0):
        depth = 50
        super(ResNet50RatioEstimator, self).__init__(
            activation=activation,
            batchnorm=batchnorm,
            channels=channels,
            convolution_bias=convolution_bias,
            dilate=dilate,
            trunk=trunk,
            trunk_dropout=trunk_dropout)



class ResNet101RatioEstimator(ResNetRatioEstimator):

    def __init__(self, activation=None,
                 batchnorm=True,
                 channels=3,
                 convolution_bias=True,
                 dilate=False,
                 trunk=[4096, 4096, 4096],
                 trunk_dropout=0.0):
        depth = 101
        super(ResNet101RatioEstimator, self).__init__(
            activation=activation,
            batchnorm=batchnorm,
            channels=channels,
            convolution_bias=convolution_bias,
            dilate=dilate,
            trunk=trunk,
            trunk_dropout=trunk_dropout)



class ResNet152RatioEstimator(ResNetRatioEstimator):

    def __init__(self, activation=None,
                 batchnorm=True,
                 channels=3,
                 convolution_bias=True,
                 dilate=False,
                 trunk=[4096, 4096, 4096],
                 trunk_dropout=0.0):
        depth = 152
        super(ResNet152RatioEstimator, self).__init__(
            activation=activation,
            batchnorm=batchnorm,
            channels=channels,
            convolution_bias=convolution_bias,
            dilate=dilate,
            trunk=trunk,
            trunk_dropout=trunk_dropout)
