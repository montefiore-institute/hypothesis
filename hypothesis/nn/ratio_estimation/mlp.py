import hypothesis as h
import torch

from hypothesis.nn import MLP
from hypothesis.nn.ratio_estimation import BaseRatioEstimator
from hypothesis.nn.util import dimensionality


def build_ratio_estimator(random_variables, **kwargs):
    activation = kwargs.get(h.default._key_activation, h.default.activation)
    dropout = kwargs.get(h.default._key_dropout, h.default.dropout)
    trunk = kwargs.get(h.default._key_trunk, h.default.trunk)
    shape_xs = (sum([dimensionality(random_variables[k]) for k in random_variables.keys()]),)
    rv_identifiers = list(random_variables.keys())

    class RatioEstimator(BaseRatioEstimator):

        def __init__(self):
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

    return RatioEstimator()
