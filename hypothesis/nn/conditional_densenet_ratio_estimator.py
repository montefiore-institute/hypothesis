import hypothesis
import torch

from hypothesis.nn import ConditionalRatioEstimator
from hypothesis.nn import DenseNet
from hypothesis.nn.util import compute_dimensionality



class ConditionalDenseNetRatioEstimator(DenseNet, ConditionalRatioEstimator):

    def __init__(self,
        shape_inputs,
        shape_outputs,
        activation=hypothesis.default.activation,
        batchnorm=True,
        channels=3,
        dense_dropout=hypothesis.default.dropout,
        depth=121, # Default DenseNet configuration
        trunk=hypothesis.default.trunk,
        trunk_activation=None,
        trunk_dropout=hypothesis.default.dropout):
        self.shape_inputs = shape_inputs
        self.dimensionality_inputs = compute_dimensionality(shape_inputs)
        self.shape_outputs = shape_outputs
        self.dimensionality_outputs = compute_dimensionality(shape_outputs)
        DenseNet.__init__(self,
            activation=activation,
            batchnorm=batchnorm,
            channels=channels,
            dense_dropout=dense_dropout,
            depth=depth,
            shape_xs=shape_outputs,
            shape_ys=(1,),
            trunk=trunk,
            trunk_activation=trunk_activation,
            trunk_dropout=trunk_dropout,
            ys_transform=None)
        ConditionalRatioEstimator.__init__(self)

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
        zs = self.network_head(outputs)
        zs = self.network_body(zs)
        zs = zs.view(-1, self.embedding_dim) # Flatten
        zs = torch.cat([zs, inputs.view(-1, self.dimensionality_inputs)], dim=1)
        log_ratios = self.network_trunk(zs)

        return log_ratios
