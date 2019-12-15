import torch

from hypothesis.nn import ConditionalRatioEstimator
from hypothesis.nn import LeNet



class ConditionalLeNetRatioEstimator(LeNet, ConditionalRatioEstimator):
    r""""""

    def __init__(self, shape_inputs, shape_outputs, activation=torch.nn.ReLU, trunk=(256, 256, 256)):
        ConditionalRatioEstimator.__init__(self)
        LeNet.__init__(self, shape_ys=shape_inputs, shape_xs=shape_outputs,
            activation=activation, trunk=trunk, transform_output=None)

    def _build_trunk(self, trunk, transform_output):
        layers = []
        layer = torch.nn.Linear(self.latent_dimensionality + self.dimensionality_ys, trunk[0])
        layers.append(layer)
        for index in range(1, len(trunk)):
            layers.append(self.activation)
            layer = torch.nn.Linear(trunk[index - 1], trunk[index])
            layers.append(layer)
        layers.append(self.activation)
        layers.append(torch.nn.Linear(trunk[-1], 1))
        self.trunk = torch.nn.Sequential(*layers)

    def forward(self, inputs, outputs):
        log_ratios = self.log_ratio(inputs, outputs)

        return log_ratios.sigmoid(), log_ratios

    def log_ratio(self, inputs, outputs):
        zs = self._forward_head(outputs)
        zs = torch.cat([zs, inputs], dim=1)
        log_ratios = self._forward_trunk(zs)

        return log_ratios
