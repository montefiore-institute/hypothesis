r"""Utilities for the catapult simulator to infer the gravitational constant.

"""

import hypothesis
import numpy as np
import torch

from hypothesis.exception import IntractableException



def Prior():
    lower = torch.tensor(1.0)
    lower = lower.to(hypothesis.accelerator)
    upper = torch.tensor(10.0)
    upper = upper.to(hypothesis.accelerator)

    return torch.distributions.uniform.Uniform(lower, upper)


def PriorExperiment(simple=False):
    if simple:
        return SimplePriorExperiment()
    else:
        return ComplexPriorExperiment()


def SimplePriorExperiment():
    r"""Coefficient of drag of the projectile and the launch angle."""
    lower = torch.tensor([0.0, 0.0])
    lower = lower.to(hypothesis.accelerator)
    upper = torch.tensor([2.0, np.pi / 2])
    upper = upper.to(hypothesis.accelerator)

    return Uniform(lower, upper)


def ComplexPriorExperiment():
    lower = torch.tensor([0.1, 1.0, 0.0, 0.0, 100.0])
    lower = lower.to(hypothesis.accelerator)
    upper = torch.tensor([2.5, 100.0, 2.0, np.pi / 2, 10000.0])
    upper = upper.to(hypothesis.accelerator)

    return Uniform(lower, upper)



class Uniform(torch.distributions.uniform.Uniform):

    r"""Used to initialize the prior over the experimental design space."""
    def __init__(self, lower, upper):
        super(Uniform, self).__init__(lower, upper)

    def log_prob(self, sample):
        return super(Uniform, self).log_prob(sample).sum()
