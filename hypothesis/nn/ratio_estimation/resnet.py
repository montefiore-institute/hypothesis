import hypothesis as h
import torch

from hypothesis.nn import MLP
from hypothesis.nn import ResNet
from hypothesis.nn.model.resnet import ResNetHead
from hypothesis.nn.model.resnet.default import batchnorm as default_batchnorm
from hypothesis.nn.model.resnet.default import channels as default_channels
from hypothesis.nn.model.resnet.default import convolution_bias as default_convolution_bias
from hypothesis.nn.model.resnet.default import depth as default_depth
from hypothesis.nn.model.resnet.default import dilate as default_dilate
from hypothesis.nn.model.resnet.default import groups as default_groups
from hypothesis.nn.model.resnet.default import in_planes as default_in_planes
from hypothesis.nn.model.resnet.default import width_per_group as default_width_per_group
from hypothesis.nn.ratio_estimation import BaseRatioEstimator
from hypothesis.nn.util import dimensionality


def build_ratio_estimator(random_variables, convolve="outputs", depth=18, **kwargs):
    if not isinstance(convolve, list):
        convolve = list([convolve])
    convolve_variables = set(convolve)
    trunk_variables = set(random_variables.keys()) - convolve_variables
    trunk_variables = list(trunk_variables)
    trunk_variables.sort()
    convolve_variables = list(convolve_variables)
    convolve_variables.sort()
    if len(convolve_variables) == 0:
        raise ValueError("No random variables to convolve have been specified (default: 'outputs').")

    class RatioEstimator(BaseRatioEstimator):

        def __init__(self,
            activation=h.default.activation,
            batchnorm=default_batchnorm,
            convolution_bias=default_convolution_bias,
            depth=depth,
            dilate=default_dilate,
            groups=default_groups,
            in_planes=default_in_planes,
            trunk_activation=None,
            trunk_dropout=h.default.dropout,
            trunk_layers=h.default.trunk,
            width_per_group=default_width_per_group):
            super(RatioEstimator, self).__init__(random_variables)
            # Construct the convolutional ResNet heads.
            self._heads = []
            for convolve_variable in convolve_variables:
                # Fetch the random variable shape
                channels = random_variables[convolve_variable][0]
                shape = random_variables[convolve_variable][1:]
                # Create the ResNet head
                head = ResNetHead(
                    activation=h.default.activation,
                    batchnorm=batchnorm,
                    channels=channels,
                    convolution_bias=convolution_bias,
                    depth=depth,
                    dilate=dilate,
                    groups=groups,
                    in_planes=in_planes,
                    shape_xs=shape,
                    width_per_group=width_per_group)
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
