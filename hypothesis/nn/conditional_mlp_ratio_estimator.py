import torch

from hypothesis.nn import BaseConditionalRatioEstimator



class ConditionalMLPRatioEstimator(BaseConditionalRatioEstimator):
    r""""""

    def __init__(self, shape_xs, shape_ys, layers=(128, 128), activation=torch.nn.ELU):
        super(ConditionalMLPRatioEstimator, self).__init__()
        self.dimensionality_xs = 1.
        self.dimensionality_ys = 1.
        self.dimensionality = self._compute_dimensionality(shape_xs, shape_ys)
        mappings = []
        mappings.append(torch.nn.Linear(self.dimensionality, layers[0]))
        for layer_index in range(len(layers) - 1):
            mappings.append(activation())
            current_layer = layers[layer_index]
            next_layer = layers[layer_index] + 1
            mappings.append(torch.nn.Linear(
                current_layer, next_layer))
        mappings.append(activation())
        self.network = torch.nn.Sequential(*mappings)

    def _compute_dimensionality(self, shape_xs, shape_ys):
        for shape_element in shape_xs:
            self.dimensionality_xs *= shape_element
        for shape_element in shape_ys:
            self.dimensionality_ys *= shape_element

        return self.dimensionality_xs + self.dimensionality_ys

    def forward(self, xs, ys):
        log_ratio = self.log_ratio(xs, ys)

        return log_ratio.sigmoid(), log_ratio

    def log_ratio(self, xs, ys):
        xs = xs.view(-1, self.dimensionality_xs)
        ys = ys.view(-1, self.dimensionality_ys)
        x = torch.cat([xs, ys], dim=1)

        return self.network(x)
