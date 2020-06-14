import hypothesis
import numpy as np
import torch

from hypothesis.diagnostic import BaseDiagnostic
from scipy.integrate import nquad



class DensityDiagnostic(BaseDiagnostic):

    def __init__(self, space, function=None, epsilon=0.1):
        super(DensityDiagnostic, self).__init__()
        self.epsilon = epsilon
        self.function = function
        self.last_result = None
        self.space = space

    def test(function=None):
        if function is None and self.function is None:
            raise ValueError("A density function needs to be provided to the method, or through the constructor.")
        elif function is None and self.function is not None:
            f = self.function
        else:
            f = function
        self.last_result = integrate.nquad(f, self.space)
        area, _ = self.last_result

        return abs(1 - area) <= self.epsilon
