"""
Method abstraction.

A method serves as an interface for doing inference in likelihood-free problems.
In these settings, one typically has a set of observations x_o that were obtained
by some experiment, and generated under some model parameters theta. This model
is typically implemented as some mechanistic computer program, i.e., a simulator.
The simulator aways one to sample from p(x | theta) in forward model. However, it
is typically intractable to obtain p(theta | x) directly from the simulator, making
inference difficult (but not impossible).
"""

import numpy as np
import torch



class Method:

    def __init__(self, simulator):
        self.simulator = simulator

    def infer(self, x_o):
        raise NotImplementedError
