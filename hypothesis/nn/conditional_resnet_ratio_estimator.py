import torch

from hypothesis.nn import ConditionalRatioEstimator
from hypothesis.nn import ResNet



class ConditionalResNetRatioEstimator(ResNet, ConditionalRatioEstimator):
    def __init__(self, depth,
                 dimensionality,
                 activation=None,
                 batchnorm=True,
                 channels=3,
                 convolution_bias=True,
                 dilate=False,
                 trunk=[4096, 4096, 4096],
                 trunk_dropout=0.0):
        self.dimensionality = int(dimensionality)
        trunk.append(1) # Final output.
        ConditionalRatioEstimator.__init__(self)
        ResNet.__init__(self,
            depth=depth,
            activation=activation,
            batchnorm=batchnorm,
            channels=channels,
            convolution_bias=convolution_bias,
            dilate=dilate,
            trunk=trunk,
            trunk_dropout=trunk_dropout)

    def _build_trunk(self):
        layers = []
        dimensionality = self.final_planes * self.block.expansion + self.dimensionality
        layers.append(torch.nn.Linear(dimensionality, self.trunk[0]))
        for i in range(1, len(self.trunk)):
            layers.append(self.activation())
            layers.append(torch.nn.Linear(self.trunk[i - 1], self.trunk[i]))
            # Check if dropout needs to be added.
            if self.trunk_dropout > 0:
                layers.append(torch.nn.Dropout(p=self.trunk_dropout))

        return torch.nn.Sequential(*layers)

    def forward(self, xs, ys):
        log_ratios = self.log_ratio(xs, ys)

        return log_ratios.sigmoid(), log_ratios

    def log_ratio(self, xs, ys):
        latents = self.network_head(ys)
        latents = self.network_body(latents)
        latents = latents.reshape(latents.size(0), -1) # Flatten
        xs = xs.reshape(-1, self.dimensionality) # Flatten inputs
        latents = torch.cat([xs, latents], dim=1)
        log_ratios = self.network_trunk(latents)

        return log_ratios



class ConditionalResNet18RatioEstimator(ConditionalResNetRatioEstimator):

    def __init__(self,
                 dimensionality,
                 activation=None,
                 batchnorm=True,
                 channels=3,
                 convolution_bias=True,
                 dilate=False,
                 trunk=[4096, 4096, 4096],
                 trunk_dropout=0.0):
        depth = 18
        super(ConditionalResNet18RatioEstimator, self).__init__(
            activation=activation,
            batchnorm=batchnorm,
            channels=channels,
            convolution_bias=convolution_bias,
            depth=depth,
            dilate=dilate,
            dimensionality=dimensionality,
            trunk=trunk,
            trunk_dropout=trunk_dropout)



class ConditionalResNet34RatioEstimator(ConditionalResNetRatioEstimator):

    def __init__(self, dimensionality,
                 activation=None,
                 batchnorm=True,
                 channels=3,
                 convolution_bias=True,
                 dilate=False,
                 trunk=[4096, 4096, 4096],
                 trunk_dropout=0.0):
        depth = 34
        super(ConditionalResNet34RatioEstimator, self).__init__(
            depth=depth,
            activation=activation,
            batchnorm=batchnorm,
            channels=channels,
            convolution_bias=convolution_bias,
            dilate=dilate,
            dimensionality=dimensionality,
            trunk=trunk,
            trunk_dropout=trunk_dropout)



class ConditionalResNet50RatioEstimator(ConditionalResNetRatioEstimator):

    def __init__(self, dimensionality,
                 activation=None,
                 batchnorm=True,
                 channels=3,
                 convolution_bias=True,
                 dilate=False,
                 trunk=[4096, 4096, 4096],
                 trunk_dropout=0.0):
        depth = 50
        super(ConditionalResNet50RatioEstimator, self).__init__(
            depth=depth,
            activation=activation,
            batchnorm=batchnorm,
            channels=channels,
            convolution_bias=convolution_bias,
            dilate=dilate,
            dimensionality=dimensionality,
            trunk=trunk,
            trunk_dropout=trunk_dropout)



class ConditionalResNet101RatioEstimator(ConditionalResNetRatioEstimator):

    def __init__(self, dimensionality,
                 activation=None,
                 batchnorm=True,
                 channels=3,
                 convolution_bias=True,
                 dilate=False,
                 trunk=[4096, 4096, 4096],
                 trunk_dropout=0.0):
        depth = 101
        super(ConditionalResNet101RatioEstimator, self).__init__(
            depth=depth,
            activation=activation,
            batchnorm=batchnorm,
            channels=channels,
            convolution_bias=convolution_bias,
            dilate=dilate,
            dimensionality=dimensionality,
            trunk=trunk,
            trunk_dropout=trunk_dropout)



class ConditionalResNet152RatioEstimator(ConditionalResNetRatioEstimator):

    def __init__(self, dimensionality,
                 activation=None,
                 batchnorm=True,
                 channels=3,
                 convolution_bias=True,
                 dilate=False,
                 trunk=[4096, 4096, 4096],
                 trunk_dropout=0.0):
        depth = 152
        super(ConditionalResNet101RatioEstimator, self).__init__(
            depth=depth,
            activation=activation,
            batchnorm=batchnorm,
            channels=channels,
            convolution_bias=convolution_bias,
            dilate=dilate,
            dimensionality=dimensionality,
            trunk=trunk,
            trunk_dropout=trunk_dropout)
