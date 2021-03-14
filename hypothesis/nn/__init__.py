r"""Specifies neural network utilities and extensions.

"""

from .model import DenseNet
from .model import MLP
from .model import ResNet

import hypothesis.nn.ratio_estimation

from hypothesis.nn.ratio_estimation import build_ratio_estimator
