import hypothesis
import hypothesis.nn
import torch

from hypothesis.nn import MultiLayeredPerceptron as MLP
from hypothesis.nn.amortized_ratio_estimation import BaseRatioEstimator
from hypothesis.nn.util import compute_dimensionality


def build_ratio_estimator(random_variables):
    class RatioEstimator(BaseRatioEstimator):

        def __init__(self,
            activation=hypothesis.default.activation,
            dropout=hypothesis.default.dropout,
            layers=hypothesis.default.trunk):
            super(RatioEstimator, self).__init__()
            self.random_variables = random_variables.keys()
            shape_xs = (sum([compute_dimensionality(shape) for shape in random_variables.values()]),)
            self.mlp = MLP(
                activation=activation,
                dropout=dropout,
                layers=layers,
                shape_xs=shape_xs,
                shape_ys=(1,),
                transform_output=None)

        def forward(self, **kwargs):
            log_ratios = self.log_ratio(**kwargs)

            return log_ratios.sigmoid(), log_ratios

        def log_ratio(self, **kwargs):
            random_variables = [kwargs[k] for k in self.random_variables]
            z = torch.cat(random_variables, dim=1)
            log_ratio = self.mlp(z)

            return log_ratio

    return RatioEstimator
