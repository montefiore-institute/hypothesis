import hypothesis
import hypothesis.nn
import torch

from hypothesis.nn import MultiLayeredPerceptron
from hypothesis.nn.amortized_ratio_estimation import BaseLikelihoodToEvidenceRatioEstimator
from hypothesis.nn.util import compute_dimensionality



class LikelihoodToEvidenceRatioEstimatorMLP(BaseLikelihoodToEvidenceRatioEstimator):

    def __init__(self,
        shape_inputs,
        shape_outputs,
        activation=hypothesis.default.activation,
        dropout=hypothesis.default.dropout,
        layers=hypothesis.default.trunk):
        super(LikelihoodToEvidenceRatioEstimatorMLP, self).__init__()
        dimensionality = compute_dimensionality(shape_inputs) + compute_dimensionality(shape_outputs)
        self.mlp = MultiLayeredPerceptron(
            shape_xs=(dimensionality,),
            shape_ys=(1,),
            activation=activation,
            dropout=dropout,
            layers=layers,
            transform_output=None)

    def log_ratio(self, inputs, outputs):
        features = torch.cat([inputs, outputs], dim=1)

        return self.mlp(features)
