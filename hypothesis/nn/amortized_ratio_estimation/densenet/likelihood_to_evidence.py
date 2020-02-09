import hypothesis
import hypothesis.nn.densenet
import torch

from hypothesis.nn import DenseNetHead
from hypothesis.nn import MultiLayeredPerceptron
from hypothesis.nn.amortized_ratio_estimation import BaseLikelihoodToEvidenceRatioEstimator
from hypothesis.nn.neuromodulation import BaseNeuromodulatedModule
from hypothesis.nn.neuromodulation import allocate_neuromodulated_activation
from hypothesis.nn.neuromodulation import list_neuromodulated_modules
from hypothesis.nn.util import compute_dimensionality



class LikelihoodToEvidenceRatioEstimatorDenseNet(BaseLikelihoodToEvidenceRatioEstimator):

    def __init__(self,
        shape_inputs,
        shape_outputs,
        activation=hypothesis.default.activation,
        batchnorm=hypothesis.nn.densenet.default.batchnorm,
        bottleneck_factor=hypothesis.nn.densenet.default.bottleneck_factor,
        channels=hypothesis.nn.densenet.default.channels,
        convolution_bias=hypothesis.nn.densenet.default.convolution_bias,
        depth=hypothesis.nn.densenet.default.depth,
        dropout=hypothesis.default.dropout,
        trunk_activation=None,
        trunk_dropout=None,
        trunk_layers=hypothesis.default.trunk):
        super(LikelihoodToEvidenceRatioEstimatorDenseNet, self).__init__()
        # Construct the convolutional DenseNet head.
        self.head = DenseNetHead(
            activation=activation,
            batchnorm=batchnorm,
            bottleneck_factor=bottleneck_factor,
            channels=channels,
            convolution_bias=convolution_bias,
            depth=depth,
            dropout=dropout,
            shape_xs=shape_outputs)
        # Compute the embedding dimensionality of the head.
        embedding_dim = self.head.embedding_dimensionality()
        # Check if custom trunk settings have been defined.
        if trunk_activation is None:
            trunk_activation = activation
        if trunk_dropout is None:
            trunk_dropout = dropout
        # Allocate the trunk.
        latent_dimensionality = compute_dimensionality(shape_inputs) + self.head.embedding_dimensionality()
        self.trunk = MultiLayeredPerceptron(
            shape_xs=(latent_dimensionality,),
            shape_ys=(1,),
            activation=trunk_activation,
            dropout=trunk_dropout,
            layers=trunk_layers,
            transform_output=None)

    def log_ratio(self, inputs, outputs):
        z_outputs = self.head(outputs)
        z = torch.cat([inputs, z_outputs], dim=1)
        log_ratios = self.trunk(z)

        return log_ratios



class LikelihoodToEvidenceRatioEstimatorNeuromodulatedDenseNet(BaseLikelihoodToEvidenceRatioEstimator):

    def __init__(self,
        shape_inputs,
        shape_outputs,
        controller_allocator,
        activation=hypothesis.default.activation,
        batchnorm=hypothesis.nn.densenet.default.batchnorm,
        bottleneck_factor=hypothesis.nn.densenet.default.bottleneck_factor,
        channels=hypothesis.nn.densenet.default.channels,
        convolution_bias=hypothesis.nn.densenet.default.convolution_bias,
        depth=hypothesis.nn.densenet.default.depth,
        dropout=hypothesis.default.dropout,
        trunk_activation=None,
        trunk_dropout=None,
        trunk_layers=hypothesis.default.trunk):
        super(LikelihoodToEvidenceRatioEstimatorNeuromodulatedDenseNet, self).__init__()
        # Allocate the neuromodulated activation.
        neuromodulated_activation = allocate_neuromodulated_activation(
            activation=activation,
            allocator=controller_allocator)
        # Construct the convolutional DenseNet head.
        self.head = DenseNetHead(
            activation=neuromodulated_activation,
            batchnorm=batchnorm,
            bottleneck_factor=bottleneck_factor,
            channels=channels,
            convolution_bias=convolution_bias,
            depth=depth,
            dropout=dropout,
            shape_xs=shape_outputs)
        # Compute the embedding dimensionality of the head.
        embedding_dim = self.head.embedding_dimensionality()
        # Check if custom trunk settings have been defined.
        if trunk_activation is None:
            trunk_activation = neuromodulated_activation
        if trunk_dropout is None:
            trunk_dropout = dropout
        # Allocate the trunk.
        self.trunk = MultiLayeredPerceptron(
            shape_xs=(self.head.embedding_dimensionality(),),
            shape_ys=(1,),
            activation=trunk_activation,
            dropout=trunk_dropout,
            layers=trunk_layers,
            transform_output=None)
        # List the neuromodulated modules.
        self.neuromodulated_modules = list_neuromodulated_modules(self)

    def log_ratio(self, inputs, outputs):
        for module in self.neuromodulated_modules:
            module.update(context=inputs)
        z = self.head(outputs)
        log_ratios = self.trunk(z)

        return log_ratios
