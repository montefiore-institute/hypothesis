"""
Adversarial Variational Optimization
"""

import cag.baseline
import cag.method



class AdversarialVariationalOptimization(Method):

    def __init__(self, simulator,
                 discriminator,
                 proposal,
                 lr_discriminator=.0001,
                 lr_proposal=.001,
                 batch_size=32,
                 r1_regularization=10.,
                 baseline=None):
        super(self, AdversarialVariationalOptimization).__init__(simulator)
        self.discriminator = discriminator
        self.proposal = proposal
        self.batch_size = batch_size
        if baseline is None:
            baseline = OptimalBaseline(discriminator)
        self._lr_discriminator = lr_discriminator
        self._lr_proposal = lr_proposal
        self._baseline = baseline
        self._real = torch.ones(self.batch_size, 1)
        self._fake = torch.zeros(self.batch_size, 1)
        self._criterion = torch.nn.BCELoss()
        self._r1 = r1_regularization
        self._o_discriminator = None
        self._o_proposal = None
        self._num_simulations = 0
        self._reset()

    def _allocate_optimizers(self):
        # Clean up the old optimizers, if any.
        if self._o_discriminator is not None:
            del self._o_discriminator
        if self._o_proposal is not None:
            del self._o_proposal
        # allocate the proposal optimizer.
        self._o_discriminator = torch.optim.RMSprop(
            self.discriminator.parameters(), lr=self._lr_discriminator
        )
        self._o_proposal = torch.optim.RMSprop(
            self.proposal.parameters(), lr=self._lr_proposal
        )

    def _reset(self):
        self._num_simulations = 0
        self._allocate_optimizers()

    def _update_critic(self, observations, thetas, x_thetas):
        raise NotImplementedError

    def _update_proposal(self, observations, thetas, x_thetas):
        raise NotImplementedError

    def infer(self, x_o, num_iterations=1000):
        self._reset()

        raise NotImplementedError
