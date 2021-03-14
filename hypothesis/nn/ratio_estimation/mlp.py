import hypothesis as h
import torch

from hypothesis.nn import MLP
from hypothesis.nn.ratio_estimation import BaseRatioEstimator
from hypothesis.nn.util import dimensionality


def build_ratio_estimator(random_variables, denominator="inputs|outputs", **kwargs):
    shapes = {}
    # Precompute shapes of random variables.
    shape_xs = 0
    for k, v in random_variables.items():
        d = dimensionality(v)
        shapes[k] =  (d,)
        shape_xs += d

    class RatioEstimator(BaseRatioEstimator):

        def __init__(self,
            activation=h.default.activation,
            dropout=h.default.dropout,
            trunk=h.default.trunk):
            super(RatioEstimator, self).__init__(denominator, random_variables)
            self._mlp = MLP(
                shape_xs=(shape_xs,),
                shape_ys=(1,),
                activation=activation,
                dropout=dropout,
                layers=trunk,
                transform_output=None)

        def log_ratio(self, **kwargs):
            tensors = [kwargs[k].view(-1, *v) for k, v in shapes.items()]
            z = torch.cat(tensors, dim=1)

            return self._mlp(z)

    return RatioEstimator
