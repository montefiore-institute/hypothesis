import hypothesis as h

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
    convolve_variables = set(list(convolve))
    trunk_variables = set(random_variables.keys()) - convolve_variables
    if len(convolve_variables) == 0:
        raise ValueError("No random variables to convolve have been specified (default: 'outputs').")

    class RatioEstimator(BaseRatioEstimator):

        def __init__(self,
            activation=h.default.activation,
            batchnorm=default_batchnorm,
            channels=default_channels,
            convolution_bias=default_convolution_bias,
            depth=depth,
            dilate=default_dilate,
            groups=default_groups,
            in_planes=default_in_planes,
            trunk_activation=None,
            trunk_dropout=h.default.dropout,
            trunk_layers=h.default.trunk,
            width_per_group=default_width_per_group):
            super(RatioEstimator, self).__init__()
            # Construct the convolutional ResNet heads.
            self._heads = []
            for convolve_variable in convolve_variables:
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
                    shape_xs=random_variables[convolve_variable],
                    width_per_group=width_per_group)
                self._heads.append(head)
            # Check if custom trunk settings have been defined.
            if trunk_activation is None:
                trunk_activation = activation
            # Construct the trunk of the network.
            self.embedding_dimensionality = sum([h.embedding_dimensionality() for h in self._heads])
            dimensionality = self.embedding_dimensionality + sum([dimensionality(random_variables[k]) for k in trunk_random_variables])
            self.trunk = MLP(
                shape_xs=(dimensionality,),
                shape_ys=(1,),
                activation=trunk_activation,
                dropout=trunk_dropout,
                layers=trunk_layers,
                transform_output=None)

        def log_ratio(self, **kwargs):
            # TODO Implement
            z_head = self.head(kwargs[convolve_variable]).view(-1, self.embedding_dimensionality)
            tensors = [kwargs[k].view(v) for k, v in trunk_random_variables.items()]
            tensors.append(z_head)
            features = torch.cat(tensors, dim=1)
            log_ratios = self.trunk(features)

            return log_ratios

    return RatioEstimator
