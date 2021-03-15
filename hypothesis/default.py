r"""Default settings in `hypothesis`.

"""

import torch


_key_activation = "activation"
activation = torch.nn.LeakyReLU
r"""Default activation function in `hypothesis`."""


_key_batch_size = "batch_size"
batch_size = 4096
r"""Default batch size."""


_key_dropout = "dropout"
dropout = 0.0
r"""Default dropout setting."""


_key_epochs = "epochs"
epochs = 100
r"""Default number of data epochs."""


output_transform = "normalize"
r"""Default output transformation for neural networks.

For 1-dimensional outputs, this is equivalent to torch.nn.Sigmoid.
Otherwise, this will reduce to torch.nn.Softmax.
"""


_key_trunk = "trunk"
trunk = (256, 256, 256)
r"""Default trunk of an MLP."""


dependent_delimiter = ','
r"""Split character indicating the dependence between random variables."""


independent_delimiter = '|'
r"""Split character indicating the independene between random variables."""


dataloader_workers = 4
r"""Default number of dataloader workers."""
