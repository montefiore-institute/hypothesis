import hypothesis
import hypothesis.nn.densenet
import torch

from hypothesis.nn import MLP
from hypothesis.nn.densenet import DenseNetHead



class DenseNet(torch.nn.Module):

    def __init__(self,
        shape_xs,
        shape_ys,
        activation=hypothesis.default.activation,
        batchnorm=hypothesis.nn.densenet.default.batchnorm,
        bottleneck_factor=hypothesis.nn.densenet.default.bottleneck_factor,
        channels=hypothesis.nn.densenet.default.channels,
        convolution_bias=hypothesis.nn.densenet.default.convolution_bias,
        depth=hypothesis.nn.densenet.default.depth,
        dropout=hypothesis.default.dropout):
        super(DenseNet, self).__init__()
        # Compute the dimensionality of the inputs.
        self.dimensionality = len(shape_xs)
        # Construct the convolutional DenseNet head.
        self.head = DenseNetHead(
            activation=activation,
            batchnorm=batchnorm,
            bottleneck_factor=bottleneck_factor,
            channels=channels,
            convolution_bias=convolution_bias,
            depth=depth,
            dropout=dropout)
        # Compute the embedding dimensionality of the head.
        embedding_dim = self.head.embedding_dimensionality()
        # Construct the trunk of the network.
        raise NotImplementedError
