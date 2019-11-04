import hypothesis
import torch

from hypothesis.engine import Procedure



class AdversarialVariationalOptimization(Procedure):
    r"""Adversarial Variational Optimization

    An implementation of arxiv.org/abs/1707.07113"""

    def __init__(self, simulator,
        discriminator,
        gamma=10.0,
        baseline=None):
        super(AdversarialVariationalOptimization, self).__init__()
        self.discriminator = discriminator
        self.simulator = simulator
        if not baseline:
            baseline = AVOBaseline(discriminator)
        self.baseline = baseline
        self.gamma = gamma

    def optimize(proposal, observations, num_steps=1):
        raise NotImplementedError
