import torch

from hypothesis.nn import BaseConditionalRatioEstimator
from hypothesis.nn import MultiLayerPerceptron as MLP



class ConditionalMLPRatioEstimator(BaseConditionalRatioEstimator):

    def __init__(self, shape_inputs, shape_outputs, layers=(128, 128), activation=torch.nn.ELU):
        super(ConditionalMLPRatioEstimator, self).__init__()
        self.dimensionality_inputs = 1
        self.dimensionality_outputs = 1
        self.dimensionality = self._compute_dimensionality(shape_inputs, shape_outputs)
        self.mlp = MLP(shape_xs=(self.dimensionality,), shape_ys=(1,),
            layers=layers, activation=activation, transform_output=None)

    def _compute_dimensionality(self, shape_inputs, shape_outputs):
        for shape_element in shape_inputs:
            self.dimensionality_inputs *= shape_element
        for shape_element in shape_outputs:
            self.dimensionality_outputs *= shape_element

        return self.dimensionality_inputs + self.dimensionality_outputs

    def forward(self, inputs, outputs):
        log_ratio = self.log_ratio(inputs, outputs)

        return log_ratio.sigmoid(), log_ratio

    def log_ratio(self, inputs, outputs):
        inputs = inputs.view(-1, self.dimensionality_inputs)
        outputs = outputs.view(-1, self.dimensionality_outputs)
        x = torch.cat([inputs, outputs], dim=1)

        return self.mlp(x)
