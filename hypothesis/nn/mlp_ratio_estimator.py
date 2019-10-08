import torch

from hypothesis.nn import BaseRatioEstimator



class MLPRatioEstimator(BaseRatioEstimator):
    r""""""

    def __init__(self, shape_xs, layers=(128, 128), activation=torch.nn.ELU):
        super(MLPRatioEstimator, self).__init__()
        self.dimensionality = 1.
        for shape_element in shape_xs:
            self.dimensionality *= shape_element
        mappings = []
        mappings.append(torch.nn.Linear(self.dimensionality, layers[0]))
        for layer_index in range(len(layers) - 1):
            mappings.append(activation())
            current_layer = layers[layer_index]
            next_layer = layers[layer_index + 1]
            mappings.append(torch.nn.Linear(
                current_layer, next_layer))
        mappings.append(activation())
        self.network = torch.nn.Sequential(*mappings)

    def forward(self, xs):
        log_ratio = self.log_ratio(xs)

        return log_ratio.sigmoid(), log_ratio


    def log_ratio(self, xs):
        xs = xs.view(-1, self.dimensionality)

        return self.network(xs)
