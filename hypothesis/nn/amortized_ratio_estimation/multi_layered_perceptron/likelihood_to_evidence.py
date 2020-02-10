import hypothesis
import hypothesis.nn
import torch

from hypothesis.nn import MultiLayeredPerceptron
from hypothesis.nn.amortized_ratio_estimation import BaseLikelihoodToEvidenceRatioEstimator
from hypothesis.nn.neuromodulation import BaseNeuromodulatedModule
from hypothesis.nn.neuromodulation import allocate_neuromodulated_activation
from hypothesis.nn.neuromodulation import list_neuromodulated_modules
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



class LikelihoodToEvidenceRatioEstimatorNeuromodulatedMLP(BaseLikelihoodToEvidenceRatioEstimator):

    def __init__(self,
        shape_outputs,
        controller_allocator,
        activation=hypothesis.default.activation,
        dropout=hypothesis.default.dropout,
        layers=hypothesis.default.trunk):
        super(LikelihoodToEvidenceRatioEstimatorNeuromodulatedMLP, self).__init__()
        # Allocate the neuromodulated activation.
        neuromodulated_activation = allocate_neuromodulated_activation(
            activation=activation,
            allocator=controller_allocator)
        # Check if the specified activation is an i
        self.mlp = MultiLayeredPerceptron(
            shape_xs=shape_outputs,
            shape_ys=(1,),
            activation=neuromodulated_activation,
            dropout=dropout,
            layers=layers,
            transform_output=None)
        # List the neuromodulated modules.
        self.neuromodulated_modules = list_neuromodulated_modules(self)

    def log_ratio(self, inputs, outputs):
        for module in self.neuromodulated_modules:
            module.update(context=inputs)

        return self.mlp(outputs)
