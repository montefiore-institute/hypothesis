import hypothesis
import hypothesis.nn
import torch

from hypothesis.nn import MultiLayeredPerceptron
from hypothesis.nn.amortized_ratio_estimation import BaseMutualInformationRatioEstimator
from hypothesis.nn.util import compute_dimensionality



class MutualInformationRatioEstimatorMLP(BaseMutualInformationRatioEstimator):

    def __init__(self,
        shape_inputs,
        shape_outputs,
        activation=hypothesis.default.activation,
        dropout=hypothesis.default.dropout,
        layers=hypothesis.default.trunk):
        super(MutualInformationRatioEstimatorMLP, self).__init__()
        dimensionality = compute_dimensionality(shape_inputs) + compute_dimensionality(shape_outputs)
        self.mlp = MultiLayeredPerceptron(
            shape_xs=(dimensionality,),
            shape_ys=(1,),
            activation=activation,
            dropout=dropout,
            layers=layers,
            transform_output=None)

    def log_ratio(self, x, y):
        features = torch.cat([x, y], dim=1)

        return self.mlp(features)
