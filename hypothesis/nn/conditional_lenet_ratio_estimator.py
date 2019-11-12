import torch

from hypothesis.nn import ConditionalRatioEstimator
from hypothesis.nn import LeNet



class ConditionalLeNetRatioEstimator(LeNet, ConditionalRatioEstimator):
    def __init__(self, shape_xs, shape_ys, activation=torch.nn.ReLU,
            batchnorm=True, trunk=(256, 256, 256)):
        super(ConditionalLeNetRatioEstimator, self).__init__(shape_xs=shape_xs,
            activation=activation, batchnorm=batchnorm, trunk=trunk)
