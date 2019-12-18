import hypothesis
import torch

from hypothesis.nn import BaseRatioEstimator
from hypothesis.nn import DenseNet



class DenseNetRatioEstimator(BaseRatioEstimator):

    def __init__(self,
        shape_xs,
        depth=121, # Default DenseNet configuration
        activation=hypothesis.default.activation,
        channels=3,
        batchnorm=True,
        bottleneck_factor=4,
        dense_dropout=hypothesis.default.dropout,
        trunk=hypothesis.default.trunk,
        trunk_dropout=hypothesis.default.dropout):
        super(DenseNetRatioEstimator, self).__init__()
        # Allocate the DenseNet model
        self.densenet = DenseNet(
            activation=activation,
            batchnorm=batchnorm,
            bottleneck_factor=bottleneck_factor,
            channels=channels,
            dense_dropout=dense_dropout,
            depth=depth,
            shape_xs=shape_xs,
            shape_ys=(1,),
            trunk=trunk,
            trunk_dropout=trunk_dropout,
            ys_transform=None)

    def forward(self, xs):
        log_ratios = self.log_ratio(xs)

        return log_ratios.sigmoid(), log_ratios

    def log_ratio(self, xs):
        return self.densenet(xs)
