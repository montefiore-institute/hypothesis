"""
Method abstraction.
"""

import numpy as np
import torch



class Method:

    def __init__(self, simulator):
        self.simulator = simulator

    def infer(self, x_o):
        raise NotImplementedError
