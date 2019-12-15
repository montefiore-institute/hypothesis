import torch

from hypothesis.nn import ConditionalRatioEstimator
from hypothesis.nn import ResNet



class ConditionalResNetRatioEstimator(ResNet, ConditionalRatioEstimator):

    def __init__(self, depth,
        shape_inputs,
        shape_outputs,
        activation=hypothesis.default.activation,
        channels=3,
        batchnorm=True,
        convolution_bias=False,
        dilate=False,
        in_planes=64,
        trunk=(512, 512, 512),
        trunk_dropout=0.0):
        # Update dimensionality data
        self.shape_inputs = shape_inputs
        self.dimensionality_inputs = 1
        for dim in shape_inputs:
            self.dimensionality_inputs *= dim
        self.shape_outputs = shape_outputs
        self.dimensionality_outputs = 1
        for dim in shape_outputs:
            self.dimensionality_outputs *= dim
        super(ConditionalRatioEstimator, self).__init__(depth=depth,
            shape_xs=shape_outputs,
            shape_ys=(1,),
            activation=activation,
            batchnorm=batchnorm,
            convolution_bias=convolution_bias,
            dilate=dilate,
            trunk=trunk,
            in_planes=in_planes,
            trunk_dropout=trunk_dropout,
            outputs_transform=None)

    def _build_trunk(self, trunk, dropout, transform_output):
        mappings = []

        # Build trunk
        mappings.append(torch.nn.Linear(self.embedding_dim + self.dimensionality_inputs, trunk[0]))
        for index in range(1, len(trunk)):
            mappings.append(self.module_activation(inplace=True))
            if dropout > 0:
                mappings.append(torch.nn.Dropout(p=dropout))
            mappings.append(torch.nn.Linear(trunk[index - 1], trunk[index]))
        # Add final fully connected mapping
        mappings.append(torch.nn.Linear(trunk[-1], 1))

        return torch.nn.Sequential(*mappings)

    def forward(self, inputs, outputs):
        r"""p(inputs|outputs)/p(inputs)"""
        log_ratios = self.log_ratio(inputs, outputs)

        return log_ratios.sigmoid(), log_ratios

    def log_ratio(self, inputs, outputs):
        r"""log p(inputs|outputs)/p(inputs)"""
        zs_head = self.network_head(outputs)
        zs_body = self.network_body(zs)
        zs = zs.view(-1, self.embedding_dim) # Flatten
        zs = torch.cat([zs, inputs.view(-1, self.dimensionality_inputs)], dim=1)
        log_ratios = self.network_trunk(zs)

        return log_ratios
