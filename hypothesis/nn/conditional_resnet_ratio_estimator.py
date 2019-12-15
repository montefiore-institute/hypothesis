import torch

from hypothesis.nn import ConditionalRatioEstimator
from hypothesis.nn import ResNet



class ConditionalResNetRatioEstimator(ResNet, ConditionalRatioEstimator):

    def __init__(self, depth,
        shape_xs,
        shape_ys,
        activation=torch.nn.ReLU,
        channels=3,
        batchnorm=True,
        convolution_bias=False,
        dilate=False,
        trunk=(512, 512, 512),
        trunk_dropout=0.0):
        super(ConditionalRatioEstimator, self).__init__(depth=depth,
            shape_xs=shape_xs,
            shape_ys=(1,),
            activation=activation,
            batchnorm=batchnorm,
            convolution_bias=convolution_bias,
            dilate=dilate,
            trunk=trunk,
            trunk_dropout=trunk_dropout,
            ys_transform=None)

    def _build_trunk(self, trunk, dropout, transform_output):
        mappings = []

        # Compute ys dimensionality
        ys_dim = 1
        for dim in self.shape_ys:
            ys_dim *= dim
        self.ys_dim = ys_dim
        # Build trunk
        mappings.append(torch.nn.Linear(self.embedding_dim + self.ys_dim, trunk[0]))
        for index in range(1, len(trunk)):
            mappings.append(self.module_activation(inplace=True))
            if dropout > 0:
                mappings.append(torch.nn.Dropout(p=dropout))
            mappings.append(torch.nn.Linear(trunk[index - 1], trunk[index]))
        # Add final fully connected mapping
        mappings.append(torch.nn.Linear(trunk[-1], 1))

        return torch.nn.Sequential(*mappings)

    def forward(self, ys, xs):
        log_ratios = self.log_ratio(xs, ys)

        return log_ratios.sigmoid(), log_ratios

    def log_ratio(self, xs, ys):
        zs_head = self.network_head(ys)
        zs_body = self.network_body(zs)
        zs = zs.view(-1, self.embedding_dim) # Flatten
        zs = torch.cat([zs, xs.view(-1, self.ys_dim)], dim=1)
        log_ratios = self.network_trunk(zs)

        return log_ratios
