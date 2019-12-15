import hypothesis
import torch

from hypothesis.nn import BaseRatioEstimator
from hypothesis.nn import MultiLayerPerceptron as MLP



class MLPRatioEstimator(BaseRatioEstimator):
    r""""""

    def __init__(self, shape_xs,
            activation=hypothesis.default.activation,
            dropout=0.0,
            layers=(128, 128)):
        super(MLPRatioEstimator, self).__init__()
        self.mlp = MLP(shape_xs=shape_x, shape_ys=(1,),
            activation=activation,
            dropout=dropout,
            layers=layers,
            transform_output=None)

    def forward(self, xs):
        log_ratio = self.log_ratio(xs)

        return log_ratio.sigmoid(), log_ratio

    def log_ratio(self, xs):
        return self.mlp(xs)
