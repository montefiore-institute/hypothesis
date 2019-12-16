r"""Hypothesis defaults.
"""

import torch
import numpy as np



activation = torch.nn.ReLU
r"""Default activation function in Hypothesis."""


output_transform = "normalize"
r"""Default output transformation for neural networks.

For 1-dimensional outputs, this is equivalent to torch.nn.Sigmoid. Otherwise, this
will reduce to torch.nn.Softmax.
"""
