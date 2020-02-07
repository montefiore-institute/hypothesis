r"""Hypothesis defaults.
"""

import hypothesis
import numpy as np
import torch



activation = torch.nn.ReLU
r"""Default activation function in Hypothesis."""

batchnorm = True
r"""Default batch normalization flag in Hypothesis."""

convolution_bias = False
r"""Add biases in convolutions by default in Hypothesis."""

channels = 3
r"""Default number of data channels (e.g., channels in images)."""

output_transform = "normalize"
r"""Default output transformation for neural networks.

For 1-dimensional outputs, this is equivalent to torch.nn.Sigmoid. Otherwise, this
will reduce to torch.nn.Softmax.
"""

trunk = (512, 512, 512)
r"""Default trunk of large convolution models such as ResNet or DenseNet."""

dropout = 0.0
r"""Default dropout setting."""
