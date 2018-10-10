"""
Adversarial Variational Optimization
"""

import torch

from cag.baseline import OptimalBaseline
from cag.method import Method
from cag.util import sample



def r1(y_hat, x):
    batch_size = x.size(0)
    grad_y_hat = torch.autograd.grad(
        outputs=y_hat.sum(),
        inputs=x,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    regularizer = grad_y_hat.pow(2).view(batch_size, -1).sum(1)

    return regularizer


class AdversarialVariationalOptimization(Method):

    def __init__(self, simulator,
                 discriminator,
                 proposal,
                 lr_discriminator=.0001,
                 lr_proposal=.001,
                 batch_size=32,
                 r1_regularization=10.,
                 baseline=None):
        super(AdversarialVariationalOptimization, self).__init__(simulator)
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
        # Allocate the optimizers.
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
        # Sample some observations.
        x_real = sample(observations, self.batch_size)
        x_fake = x_thetas
        x_real.requires_grad = True
        y_real = self.discriminator(x_real)
        y_fake = self.discriminator(x_fake)
        loss = (self._criterion(y_real, self._real) + self._criterion(y_fake, self._fake)) / 2.
        loss = loss + self._r1 * r1(y_real, x_real).mean()
        loss.backward()
        self._o_discriminator.step()

    def _update_proposal(self, observations, thetas, x_thetas):
        log_probabilities = self.proposal.log_prob(thetas)
        gradients = []
        for log_p in log_probabilities:
            gradient = torch.autograd.grad(log_p, self.proposal.parameters(), create_graph=True)
            gradients.append(gradient)
        gradient_U = []
        with torch.no_grad():
            # Allocate buffer for all parameters in the proposal.
            for p in gradients[0]:
                gradient_U.append(torch.zeros_like(p))
            p_thetas = self._baseline.apply(gradients, x_thetas)
            for index, gradient in enumerate(gradients):
                p_theta = p_thetas[index]
                for pg_index, pg in enumerate(gradient):
                    pg_theta = p_theta[pg_index].squeeze()
                    gradient_U[pg_index] += -pg_theta * pg
            # Average out U.
            for p in gradient_U:
                p /= self.batch_size
            for index, p in enumerate(self.proposal.parameters()):
                p.grad = gradient_U[index].expand(p.size())
        self._o_proposal.step()
        self.proposal.fix()

    def _sample(self):
        thetas = self.proposal.sample(self.batch_size)
        thetas, x_thetas = self.simulator(thetas)
        self._num_simulations += self.batch_size

        return thetas, x_thetas

    def step(self, x_o):
        thetas, x_thetas = self._sample()
        self._update_critic(x_o, thetas, x_thetas)
        self._update_proposal(x_o, thetas, x_thetas)

    def infer(self, x_o, num_steps=1000):
        self._reset()

        for iteration in range(num_steps):
            self.step(x_o)

        return self.proposal.clone()
