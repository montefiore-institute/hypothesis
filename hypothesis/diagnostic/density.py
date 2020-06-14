import hypothesis
import numpy as np
import torch

from hypothesis.diagnostic import BaseDiagnostic
from scipy.integrate import nquad



class DensityDiagnostic(BaseDiagnostic):

    def __init__(self, space, density, epsilon=0.1):
        super(DensityDiagnostic, self).__init__()
        self.density = density
        self.epsilon = epsilon
        self.last_result = None
        self.space = space

    def test(**kwargs):
        self.last_result = integrate.nquad(self.pdf, self.space)
        area, _ = self.last_result

        return abs(1 - area) <= self.epsilon
