r"""Utilities for the catapult simulator to infer the gravitational constant.

"""

import torch

from hypothesis.exception import IntractableException



def Prior():
    raise NotImplementedError


def PriorExperiment():
    raise NotImplementedError


def Truth():
    raise NotImplementedError


def log_likelihood(theta, x):
    raise IntractableException
