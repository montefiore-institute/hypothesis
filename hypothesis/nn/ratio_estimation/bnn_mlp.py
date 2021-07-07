import hypothesis as h
import torch

from hypothesis.nn import BNNMLP
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
            self._mlp = BNNMLP(
                shape_xs=(shape_xs,),
                shape_ys=(1,),
                activation=activation,
                layers=trunk,
                transform_output=None)

        # The log ratio obtained with 1 set of weights randomly sampled
        def log_ratio(self, **kwargs):
            tensors = [kwargs[k].view(-1, *v) for k, v in shapes.items()]
            z = torch.cat(tensors, dim=1)

            return self._mlp(z)

        def kl_loss(self):
            return self._mlp.kl_loss()

    return RatioEstimator
