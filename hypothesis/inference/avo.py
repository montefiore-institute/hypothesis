"""
Adversarial Variational Optimization
"""

import torch

from hypothesis.engine import event
from hypothesis.inference import SimulatorMethod
from hypothesis.inference.baseline import AVOBaseline
from hypothesis.util import sample



def r1_regularization(y_hat, x):
    batch_size = x.size(0)
    grad_y_hat = torch.autograd.grad(
        outputs=y_hat.sum(),
        inputs=x,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    regularizer = grad_y_hat.pow(2).view(batch_size, -1).sum()

    return regularizer


class AdversarialVariationalOptimization(SimulatorMethod):

    KEY_STEPS = "steps"

    def __init__(self, simulator,
                 discriminator,
                 proposal,
                 lr_discriminator=.0001,
                 lr_proposal=.001,
                 batch_size=32,
                 gamma=10.,
                 baseline=None):
        super(AdversarialVariationalOptimization, self).__init__(simulator)
        # Initialize the state of the procedure.
        self.discriminator = discriminator
        self.proposal = proposal.clone()
        self.batch_size = batch_size
        if not baseline:
            baseline = AVOBaseline(discriminator)
        self.baseline = baseline
        self._lr_discriminator = lr_discriminator
        self._lr_proposal = lr_proposal
        self._real = torch.ones(self.batch_size, 1)
        self._fake = torch.zeros(self.batch_size, 1)
        self._criterion = torch.nn.BCELoss()
        self._gamma = gamma
        self._o_discriminator = None
        self._o_proposal = None
        self._num_simulations = 0
        self._reset()
        # Add AVO specific events.
        event.add_event("avo_simulation_start")
        event.add_event("avo_simulation_end")
        event.add_event("avo_update_discriminator_start")
        event.add_event("avo_update_discriminator_end")
        event.add_event("avo_update_proposal_start")
        event.add_event("avo_update_proposal_end")

    def _reset(self):
        self._num_simulations = 0
        self._allocate_optimizers()

    def _allocate_optimizers(self):
        # Clean up the old optimizers, if any:
        if self._o_discriminator:
            del self._o_discriminator
        if self._o_proposal:
            del self._o_proposal
        # Allocate the new optimizers.
        self._o_discriminator = torch.optim.RMSprop(
            self.discriminator.parameters(), lr=self._lr_discriminator
        )
        self._o_proposal = torch.optim.RMSprop(
            self.proposal.parameters(), lr=self._lr_proposal
        )

    def _update_discriminator(self, observations, thetas, x_thetas):
        self.fire_event(event.avo_update_discriminator_start)
        x_real = sample(observations, self.batch_size)
        x_fake = x_thetas
        x_real.requires_grad = True
        y_real = self.discriminator(x_real)
        y_fake = self.discriminator(x_fake)
        loss = (self._criterion(y_real, self._real) + self._criterion(y_fake, self._fake)) / 2.
        loss = loss + self._gamma * r1_regularization(y_real, x_real).mean()
        self._o_discriminator.zero_grad()
        loss.backward()
        self._o_discriminator.step()
        x_real.requires_grad = False
        self.fire_event(event.avo_update_discriminator_end)

    def _update_proposal(self, observations, thetas, x_thetas):
        self.fire_event(event.avo_update_proposal_start)
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
            p_thetas = self.baseline.apply(gradients=gradients, x=x_thetas)
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
        self.fire_event(event.avo_update_proposal_end)

    def _sample(self):
        self.fire_event(event.avo_simulation_start)
        thetas = self.proposal.sample(self.batch_size)
        thetas, x_thetas = self.simulator(thetas)
        self._num_simulations += self.batch_size
        self.fire_event(event.avo_simulation_end)

        return thetas, x_thetas

    def step(self, observations):
        thetas, x_thetas = self._sample()
        self._update_discriminator(observations, thetas, x_thetas)
        self._update_proposal(observations, thetas, x_thetas)

    def procedure(self, observations, **kwargs):
        self._reset()
        num_steps = int(kwargs[self.KEY_STEPS])
        for iteration in range(num_steps):
            self.fire_event(event.iteration_start)
            self.step(observations);
            self.fire_event(event.iteration_end)

        return self.proposal.clone()
