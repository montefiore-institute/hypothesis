import hypothesis
import torch

from hypothesis.nn import BaseConditionalRatioEstimator
from hypothesis.nn import MultiLayerPerceptron as MLP
from hypothesis.nn.util import compute_dimensionality



class ConditionalMLPRatioEstimator(BaseConditionalRatioEstimator):

    def __init__(self, shape_inputs, shape_outputs,
            activation=hypothesis.default.activation,
            dropout=0.0,
            layers=(128, 128)):
        super(ConditionalMLPRatioEstimator, self).__init__()
        self.dimensionality_inputs = compute_dimensionality(shape_inputs)
        self.dimensionality_outputs = compute_dimensionality(shape_outputs)
        self.mlp = MLP(
            shape_xs=(dimensionality_inputs + dimensionality_outputs,),
            shape_ys=(1,),
            activation=activation,
            dropout=dropout,
            layers=layers,
            transform_output=None)

    def forward(self, inputs, outputs):
        log_ratio = self.log_ratio(inputs, outputs)

        return log_ratio.sigmoid(), log_ratio

    def log_ratio(self, inputs, outputs):
        inputs = inputs.view(-1, self.dimensionality_inputs)
        outputs = outputs.view(-1, self.dimensionality_outputs)
        x = torch.cat([inputs, outputs], dim=1)

        return self.mlp(x)
