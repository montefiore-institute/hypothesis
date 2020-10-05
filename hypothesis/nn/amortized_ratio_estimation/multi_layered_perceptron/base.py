import hypothesis
import hypothesis.nn
import torch

from hypothesis.nn import MultiLayeredPerceptron as MLP
from hypothesis.nn.amortized_ratio_estimation import BaseRatioEstimator
from hypothesis.nn.util import compute_dimensionality


def build_ratio_estimator(random_variables):
    # Flatten the shapes of the random variables
    for k in random_variables.keys():
        shape = random_variables[k]
        flattened_shape = (compute_dimensionality(shape),)
        random_variables[k] = flattened_shape

    class RatioEstimator(BaseRatioEstimator):

        def __init__(self,
            activation=hypothesis.default.activation,
            dropout=hypothesis.default.dropout,
            layers=hypothesis.default.trunk):
            super(RatioEstimator, self).__init__()
            shape_xs = (sum([compute_dimensionality(shape) for shape in random_variables.values()]),)
            self.mlp = MLP(
                activation=activation,
                dropout=dropout,
                layers=layers,
                shape_xs=shape_xs,
                shape_ys=(1,),
                transform_output=None)

        def log_ratio(self, **kwargs):
            tensors = [kwargs[k].view(-1, random_variables[k][0]) for k in random_variables]
            z = torch.cat(tensors, dim=1)
            log_ratios = self.mlp(z)

            return log_ratios

    return RatioEstimator
