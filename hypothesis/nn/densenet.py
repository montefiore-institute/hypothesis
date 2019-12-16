import hypothesis
import numpy as np
import torch

from hypothesis.nn.util import allocate_output_transform
from hypothesis.nn.util import compute_dimensionality



class DenseNet(torch.nn.Module):

    def __init__(self,
        config,
        shape_xs,
        shape_ys=(1,),
        activation=hypothesis.default.activation,
        batchnorm=True,
        bottleneck_factor=4,
        channels=3,
        convolution_bias=False,
        dense_dropout=0.0,
        growth_rate=32,
        in_planes=64,
        trunk=(512, 512, 512),
        trunk_dropout=0.0,
        ys_transform=hypothesis.default.output_transform):
        super(DenseNet, self).__init__()
        # Infer the dimensionality from the input shape.
        dimensionality = len(shape_xs)

    def forward(self):
        raise NotImplementedError
