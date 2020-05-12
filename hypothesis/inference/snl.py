"""Sequential Neural Likelihood

An implementation of Sequential Neural Likelihood (SNL)

https://arxiv.org/abs/1805.07226
"""

import hypothesis
import hypothesis.inference.mcmc
import numpy as np
import torch

from hypothesis.engine import Procedure
from hypothesis.summary.mcmc import Chain
from torch.multiprocessing import Pool
