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
        log_ratio = self.resnet(xs)

        return log_ratio.sigmoid(), log_ratio

    def log_ratio(self, xs):
        log_ratio = self.resnet(xs)

        return log_ratio
