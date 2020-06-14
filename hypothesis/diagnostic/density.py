import hypothesis
import numpy as np
import torch

from hypothesis.diagnostic import BaseDiagnostic
from scipy.integrate import nquad



class DensityDiagnostic(BaseDiagnostic):

    def __init__(self, space, pdf, epsilon=0.1):
        super(DensityDiagnostic, self).__init__()
        self.epsilon = epsilon
        self.pdf = pdf
        self.result = None
        self.space = space

    def test(**kwargs):
        self.result = integrate.nquad(self.pdf, self.space)
        area, _ = self.result

        return abs(1 - area) <= self.epsilon
