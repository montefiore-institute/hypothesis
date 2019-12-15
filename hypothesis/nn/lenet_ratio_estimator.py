import torch

from hypothesis.nn import BaseRatioEstimator
from hypothesis.nn import LeNet


class LeNetRatioEstimator(BaseRatioEstimator):
    r""""""

    def __init__(self, shape_xs, trunk=(256, 256, 256), activation=hypothesis.default.activation):
        super(LeNetRatioEstimator, self).__init__()
        self.lenet = LeNet(shape_xs=shape_x, shape_ys=(1,),
            trunk=trunk, activation=activation, transform_output=None)

    def forward(self, xs):
        log_ratio = self.log_ratio(xs)

        return log_ratio.sigmoid(), log_ratio

    def log_ratio(self, xs):
        return self.lenet(xs)
