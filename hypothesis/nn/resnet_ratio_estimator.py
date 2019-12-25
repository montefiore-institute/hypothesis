import hypothesis
import torch

from hypothesis.nn import BaseRatioEstimator
from hypothesis.nn import ResNet



class ResNetRatioEstimator(BaseRatioEstimator):

    def __init__(self,
        shape_xs,
        depth=18, # Default ResNet depth
        activation=hypothesis.default.activation,
        channels=3,
        batchnorm=True,
        convolution_bias=False,
        dilate=False,
        trunk=(512, 512, 512),
        trunk_activation=None,
        trunk_dropout=0.0):
        super(BaseRatioEstimator, self).__init__()
        # Allocate the ResNet model
        self.resnet = ResNet(depth=depth,
            shape_xs=shape_xs,
            shape_ys=(1,),
            activation=activation,
            batchnorm=batchnorm,
            channels=channels,
            convolution_bias=convolution_bias,
            dilate=dilate,
            trunk=trunk,
            trunk_activation=trunk_activation,
            trunk_dropout=trunk_dropout,
            ys_transform=None)

    def forward(self, xs):
        log_ratios = self.log_ratio(xs)

        return log_ratios.sigmoid(), log_ratios

    def log_ratio(self, xs):
        return self.resnet(xs)
