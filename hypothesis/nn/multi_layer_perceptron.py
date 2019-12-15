r"""Multilayer Perceptron
"""

import hypothesis
import torch



class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, shape_xs, shape_ys,
        activation=hypothesis.default.activation,
        dropout=0.0,
        layers=(128, 128),
        transform_output="normalize"):
        super(MultiLayerPerceptron, self).__init__()
        mappings = []
        dropout = float(dropout)
        # Allocate input mapping
        mappings.append(torch.nn.Linear(
            self.compute_dimensionality(shape_xs), layers[0]))
        # Allocate internal network structure
        for index in range(1, len(layers)):
            mappings.append(self._make_layer(activation, dropout,
                layers[index - 1], layers[index]))
        # Allocate tail
        ys_dimensionality = self.compute_dimensionality(shape_ys)
        mappings.append(activation(inplace=True))
        mappings.append(layers[-1], ys_dimensionality)
        operation = None
        if transform_output is "normalize":
            if ys_dimensionality > 1:
                operation = torch.nn.Softmax(dim=0)
            else:
                operation = torch.nn.Sigmoid()
        elif transform_output is not None:
            operation = transform_output()
        if operation is not None:
            self.mapping.append(operation)
        # Allocate sequential mapping
        self.mapping = torch.nn.Sequential(*mappings)

    def _make_layer(self, activation, dropout, num_a, num_b):
        mappings = []

        mappings.append(activation(inplace=True))
        if dropout > 0:
            mappings.append(torch.nn.Dropout(p=dropout))
        mappings.append(torch.nn.Linear(num_a, num_b))

        return torch.nn.Sequential(*mappings)


    def forward(self, xs):
        return self.mapping(xs)

    @staticmethod
    def compute_dimensionality(shape):
        dimensionality = 1
        for dim in shape:
            dimensionality *= dim

        return dimensionality
