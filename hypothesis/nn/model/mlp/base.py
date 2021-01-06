r"""Base model of a multi-layered perceptron (MLP).

"""

import hypothesis as h
import torch

from hypothesis.nn.util import allocate_output_transform
from hypothesis.nn.util import dimensionality


class MLP(torch.nn.Module):

    def __init__(self,
                 shape_xs,
                 shape_ys,
                 activation=h.default.activation,
                 dropout=h.default.dropout,
                 layers=h.default.trunk,
                 transform_output=h.default.output_transform):
        r"""Initializes a multi-layered perceptron (MLP).

        :param shape_xs: A tuple describing the shape of the MLP inputs.
        :param shape_ys: A tuple describing the shape of the MLP outputs.
        :param activation: An allocator which, when called,
                           returns a :mod:`torch` activation.
        :param dropout: Dropout rate.
        :param transform_output: Output transformation.

        :rtype: :class:`hypothesis.nn.model.mlp.MLP`
        """
        super(MLP, self).__init__()
        dropout = float(dropout)  # Ensure a float
        # Set the dimensionality of the MLP inputs and outputs.
        self._dimensionality_xs = dimensionality(shape_xs)
        self._dimensionality_ys = dimensionality(shape_ys)
        # Construct the forward architecture of the MLP.
        mappings = []
        mappings.append(torch.nn.Linear(self._dimensionality_xs, layers[0]))
        for index in range(1, len(layers)):
            layer = self._make_layer(activation, dropout,
                                     layers[index - 1], layers[index])
            mappings.append(layer)
        mappings.append(activation())
        mappings.append(torch.nn.Linear(layers[-1], self._dimensionality_ys))
        operation = allocate_output_transform(transform_output,
                                              self._dimensionality_ys)
        if operation is not None:
            mappings.append(operation)
        self._mapping = torch.nn.Sequential(*mappings)

    def _make_layer(self, activation, dropout, num_a, num_b):
        r"""Creates a layer in the MLP based on the specified activation function,
        dropout rate, and dimensionality of the previous and next layer."""
        mappings = []

        mappings.append(activation())
        if dropout > 0:
            mappings.append(torch.nn.Dropout(p=dropout))
        mappings.append(torch.nn.Linear(num_a, num_b))

        return torch.nn.Sequential(*mappings)

    def forward(self, xs):
        y = self._mapping(xs)

        return y
