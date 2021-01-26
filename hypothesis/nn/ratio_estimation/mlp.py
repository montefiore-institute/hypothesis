import hypothesis as h
import torch

from hypothesis.nn import MLP
from hypothesis.nn.ratio_estimation import BaseRatioEstimator
from hypothesis.nn.util import dimensionality


def build_ratio_estimator(random_variables, **kwargs):
    shape_xs = (sum([dimensionality(random_variables[k]) for k in random_variables.keys()]),)
    rv_identifiers = list(random_variables.keys())
    rv_identifiers.sort()

    class RatioEstimator(BaseRatioEstimator):

        def __init__(self,
            activation=h.default.activation,
            dropout=h.default.dropout,
            trunk=h.default.trunk):
            super(RatioEstimator, self).__init__(random_variables)
            self._mlp = MLP(
                shape_xs=shape_xs,
                shape_ys=(1,),
                activation=activation,
                dropout=dropout,
                layers=trunk,
                transform_output=None)

        def log_ratio(self, **kwargs):
            z = torch.cat([kwargs[k] for k in rv_identifiers], dim=1)  # Assume shapes are correct

            return self._mlp(z)

    return RatioEstimator
