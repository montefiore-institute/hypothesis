"""
Method abstraction.
"""

import numpy as np
import torch

from cag.engine import Module



class Method(Module):

    def __init__(self, simulator):
        super(Method, self).__init__()
        self.simulator = simulator

    def infer(self, x_o):
        raise NotImplementedError
