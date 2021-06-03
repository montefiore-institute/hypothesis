r"""Implementation of Adversarial Variational Optimization

"""

import hypothesis as h
import numpy as np

from hypothesis.engine import Procedure
from .baseline import MeanBaseline


class AdversarialVariationalOptimization(Procedure):

    def __init__(self,
        discriminator,
        proposal,
        simulator,
        baseline=None,
        lr_discriminator=0.0001,
        lr_proposal=0.001,
        batch_size=32,
        r1_regularization=10.0):
        super(AdversarialVariationalOptimization, self).__init__()
        # Public properties
        self.discriminator = discriminator
        self.simulator = simulator
        self.proposal = proposal
        # Private properties
        if baseline is None:
            baseline = MeanBaseline(discriminator)
        self._baseline = baseline
        self._batch_size = batch_size
        self._criterion = torch.nn.BCELoss()
        self._ones = torch.ones(batch_size, 1)
        self._optimizer_d = None
        self._optimizer_p = None
        self._r1 = r1_regularization
        self._zeros = torch.zeros(batch_size, 1)
        self._allocate_optimizers(lr_discriminator, lr_proposal)

    def _allocate_optimizers(self, lr_d, lr_p):
        # Clean up old optimizers
        if self._optimizer_d is None:
            del self._optimizer_d
        if self._optimizer_p is None:
            del self._optimizer_p
        # Allocate the optimizers
        self._optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self._lr_discriminator)
        self._optimizer_p = torch.optim.Adam(
            self.proposal.parameters(),
            lr=self._lr_proposal)

    def _register_events(self):
        pass  # TODO Implement

    def infer(self, observables, steps=10000):
        for step in range(steps):
            self.step(observables)

        return self.proposal
