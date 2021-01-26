import hypothesis as h
import numpy as np
import torch

from hypothesis.nn import DenseNet
from hypothesis.nn import MLP
from hypothesis.nn.model.densenet import DenseNetHead
from hypothesis.nn.model.densenet.default import batchnorm as default_batchnorm
from hypothesis.nn.model.densenet.default import bottleneck_factor as default_bottleneck_factor
from hypothesis.nn.model.densenet.default import channels as default_channels
from hypothesis.nn.model.densenet.default import convolution_bias as default_convolution_bias
from hypothesis.nn.model.densenet.default import depth as default_depth
from hypothesis.nn.ratio_estimation import BaseRatioEstimator
from hypothesis.nn.util import dimensionality



def build_ratio_estimator(random_variables, convolve="outputs", depth=121, **kwargs):
    if not isinstance(convolve, list):
        convolve = list([convolve])
    if not isinstance(depth, list):
        depths = list([depth] * len(convolve))
    else:
        depths = depth
    depths = np.array(depths)
    assert len(depths) == len(convolve)
    convolve_variables = set(convolve)
    trunk_variables = set(random_variables.keys()) - convolve_variables
    trunk_variables = list(trunk_variables)
    trunk_variables.sort()
    convolve_variables = np.array(list(convolve_variables))
    indices = convolve_variables.argsort()
    convolve_variables = convolve_variables[indices]
    depths = depths[indices]
    if len(convolve_variables) == 0:
        raise ValueError("No random variables to convolve have been specified (default: 'outputs').")

    class RatioEstimator(BaseRatioEstimator):

        def __init__(self,
                 activation=h.default.activation,
                 batchnorm=default_batchnorm,
                 bottleneck_factor=default_bottleneck_factor,
                 channels=default_channels,
                 convolution_bias=default_convolution_bias,
                 depth=default_depth,
                 dropout=h.default.dropout,
                 trunk_activation=None,
                 trunk_dropout=h.default.dropout,
                 trunk_layers=h.default.trunk):
            super(RatioEstimator, self).__init__(random_variables)
            # Construct the convolutional ResNet heads.
            self._heads = []
            for index, convolve_variable in enumerate(convolve_variables):
                # Fetch the random variable shape
                channels = random_variables[convolve_variable][0]
                shape = random_variables[convolve_variable][1:]
                # Create the DenseNet head
                head = DenseNetHead(
                    activation=activation,
                    batchnorm=batchnorm,
                    bottleneck_factor=bottleneck_factor,
                    channels=channels,
                    convolution_bias=convolution_bias,
                    depth=depth,
                    dropout=dropout,
                    shape_xs=shape)
                self._heads.append(head)
            # Check if custom trunk settings have been defined.
            if trunk_activation is None:
                trunk_activation = activation
            # Compute the embedding dimensionalities of every head.
            self._embedding_dimensionalities = []
            for head in self._heads:
                dim = head.embedding_dimensionality()
                self._embedding_dimensionalities.append(dim)
            # Construct the trunk of the network.
            self._embedding_dimensionality = sum(self._embedding_dimensionalities)
            total_dimensionality = self._embedding_dimensionality + sum([dimensionality(random_variables[k]) for k in trunk_variables])
            self._trunk = MLP(
                shape_xs=(total_dimensionality,),
                shape_ys=(1,),
                activation=trunk_activation,
                dropout=trunk_dropout,
                layers=trunk_layers,
                transform_output=None)

        def log_ratio(self, **kwargs):
            # Compute the embedding of all heads.
            z = []
            for index, head in enumerate(self._heads):
                random_variable = convolve_variables[index]
                head_latent = head(kwargs[random_variable]).view(-1, self._embedding_dimensionalities[index])
                z.append(head_latent)
            # Join the remaining random variables
            for random_variable in trunk_variables:
                shape = random_variables[random_variable]
                z.append(kwargs[random_variable].view(-1, *shape))
            # Concatonate the tensors
            z = torch.cat(z, dim=1)

            return self._trunk(z)

    return RatioEstimator
