r"""Base model of a Bayesian multi-layered perceptron (MLP).

"""

import hypothesis as h
import torch
import torch.nn.functional as F
import numpy as np

from hypothesis.nn.util import allocate_output_transform
from hypothesis.nn.util import dimensionality
from torch.nn import Parameter


class BNNLinear(torch.nn.Module):
    def __init__(self, in_shape, out_shape, prior_w_mu, prior_w_sigma, prior_bias_mu, prior_bias_sigma):
        super(BNNLinear, self).__init__()
        self.prior_w_mu = torch.Tensor([prior_w_mu])
        self.prior_w_sigma = torch.Tensor([prior_w_sigma])
        self.prior_bias_mu = torch.Tensor([prior_bias_mu])
        self.prior_bias_sigma = torch.Tensor([prior_bias_sigma])

        self.w_mu = Parameter(torch.normal(torch.full((out_shape, in_shape), prior_w_mu), torch.full((out_shape, in_shape), prior_w_sigma)))
        self.w_log_sigma = Parameter(torch.full((out_shape, in_shape), np.log(prior_w_sigma)))

        self.bias_mu = Parameter(torch.normal(torch.full((out_shape,), prior_bias_mu), torch.full((out_shape,), prior_bias_sigma)))
        self.bias_log_sigma = Parameter(torch.full((out_shape,), np.log(prior_bias_sigma)))

    def forward(self, xs):
        u = torch.normal(torch.full(self.w_mu.shape, 0.), torch.full(self.w_mu.shape, 1.))
        weight = u * torch.exp(self.w_log_sigma) + self.w_mu

        v = torch.normal(torch.full(self.bias_mu.shape, 0.), torch.full(self.bias_mu.shape, 1.))
        bias = v * torch.exp(self.bias_log_sigma) + self.bias_mu

        return F.linear(xs, weight, bias)

    def compute_kl(self, p_mu, log_p_sigma, q_mu, log_q_sigma):
        return (log_q_sigma - log_p_sigma + (torch.exp(log_p_sigma - log_q_sigma)).pow(2)/2 + ((p_mu - q_mu)/torch.exp(log_q_sigma)).pow(2)/2 - 0.5).sum()

    def kl_loss(self):
        weight_kl = self.compute_kl(self.w_mu, self.w_log_sigma, self.prior_w_mu, torch.log(self.prior_w_sigma))
        bias_kl = self.compute_kl(self.bias_mu, self.bias_log_sigma, self.prior_bias_mu, torch.log(self.prior_bias_sigma))

        return weight_kl + bias_kl

# Modify
class BNNMLP(torch.nn.Module):

    def __init__(self,
                 shape_xs,
                 shape_ys,
                 activation=h.default.activation,
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
        super(BNNMLP, self).__init__()
        # Set the dimensionality of the MLP inputs and outputs.
        self._dimensionality_xs = dimensionality(shape_xs)
        self._dimensionality_ys = dimensionality(shape_ys)
        # Construct the forward architecture of the MLP.
        mappings = []
        mappings.append(BNNLinear(self._dimensionality_xs, layers[0], 0., .05, 0., .05))
        for index in range(1, len(layers)):
            layer = self._make_layer(activation, layers[index - 1], layers[index])
            mappings.append(layer)
        mappings.append(activation())
        mappings.append(BNNLinear(layers[-1], self._dimensionality_ys, 0., .05, 0., .05))
        operation = allocate_output_transform(transform_output,
                                              self._dimensionality_ys)
        if operation is not None:
            mappings.append(operation)
        self._mapping = torch.nn.Sequential(*mappings)

    def _make_layer(self, activation, num_a, num_b):
        r"""Creates a layer in the MLP based on the specified activation function,
        dropout rate, and dimensionality of the previous and next layer."""
        mappings = []

        mappings.append(activation())
        mappings.append(BNNLinear(num_a, num_b, 0., .05, 0., .05))

        return torch.nn.Sequential(*mappings)

    def forward(self, xs):
        y = self._mapping(xs)

        return y

    def kl_loss(self):
        kl = 0.
        for layer in self._mapping.children():
            if isinstance(layer, BNNLinear):
                kl = kl + layer.kl_loss()

        return kl

