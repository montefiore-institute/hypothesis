r"""Hypothesis defaults.
"""

import hypothesis
import numpy as np
import torch



activation = torch.nn.ReLU
r"""Default activation function in Hypothesis."""

batch_size = 128
r"""Default batch size."""

dropout = 0.0
r"""Default dropout setting."""

output_transform = "normalize"
r"""Default output transformation for neural networks.

For 1-dimensional outputs, this is equivalent to torch.nn.Sigmoid. Otherwise, this
will reduce to torch.nn.Softmax.
"""

trunk = (512, 512, 512)
r"""Default trunk of large convolution models such as ResNet or DenseNet."""

dependent_delimiter = ','
r"""Split character indicating the dependence between random variables."""

independent_delimiter = '|'
r"""Split character indicating the independene between random variables."""
