import hypothesis
import numpy as np
import torch

from hypothesis.diagnostic import BaseDiagnostic
from scipy.integrate import nquad



class DensityDiagnostic(BaseDiagnostic):

    def __init__(self, space, epsilon=0.1):
        super(DensityDiagnostic, self).__init__()
        self.epsilon = epsilon
        self.results = []
        self.space = space

    def reset(self):
        self.results = []

    def test(self, function):
        area, error = integrate.nquad(function, self.space)
        passed = abs(1 - area) <= self.epsilon
        self.results.append(result)

        return passed
