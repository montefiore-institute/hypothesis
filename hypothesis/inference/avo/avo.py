r"""Implementation of Adversarial Variational Optimization

"""

import hypothesis as h
import numpy as np
import torch

from hypothesis.engine import Procedure
from .baseline import MeanBaseline


class AdversarialVariationalOptimization(Procedure):

    def __init__(self,
        discriminator,
        proposal,
        simulator,
        baseline=None,
        lr_discriminator=0.0001,
        lr_proposal=0.01,
        weight_decay=0.0,
        batch_size=32):
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
        self._zeros = torch.zeros(batch_size, 1)
        self._allocate_optimizers(
            lr_d=lr_discriminator,
            lr_p=lr_proposal,
            weight_decay=weight_decay)

    def _allocate_optimizers(self, lr_d, lr_p, weight_decay):
        # Clean up old optimizers
        if self._optimizer_d is None:
            del self._optimizer_d
        if self._optimizer_p is None:
            del self._optimizer_p
        # Allocate the optimizers
        self._optimizer_d = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=lr_d,
            weight_decay=weight_decay)
        self._optimizer_p = torch.optim.AdamW(
            self.proposal.parameters(),
            lr=lr_p)

    def _register_events(self):
        self.register_event("fit_complete")
        self.register_event("fit_start")
        self.register_event("step_complete")
        self.register_event("step_start")

    def _update_critic(self, observables, outputs):
        # Sample random observables and prepare
        indices = np.random.choice(np.arange(len(observables)), replace=False, size=self._batch_size)
        x_real = observables[indices, :].detach()
        x_fake = outputs
        # Update the discriminator
        self._optimizer_d.zero_grad()
        y_real = self.discriminator(x_real)
        y_fake = self.discriminator(x_fake)
        loss = self._criterion(y_real, self._ones) + self._criterion(y_fake, self._zeros)
        loss.backward()
        self._optimizer_d.step()

    def _update_proposal(self, inputs, outputs):
        self._optimizer_p.zero_grad()
        # Compute the gradients of the log probabilities.
        gradients = []
        log_probabilities = self.proposal.log_prob(inputs)
        for log_p in log_probabilities:
            gradient = torch.autograd.grad(log_p, self.proposal.parameters(), create_graph=True)
            gradients.append(gradient)
        # Compute the REINFORCE gradient wrt the model parameters.
        gradient_U = []
        with torch.no_grad():
            # Allocate a buffer for all parameters in the proposal.
            for p in self.proposal.parameters():
                gradient_U.append(torch.zeros_like(p))
            # Apply a baseline for variance reduction in the theta grads.
            p_thetas = self._baseline.apply(observables=outputs, gradients=gradients).squeeze()
            # Compute the REINFORCE gradient.
            for index, gradient in enumerate(gradients):
                p_theta = p_thetas[index]
                for p_index, p_gradient in enumerate(gradient):
                    gradient_U[p_index] += -p_theta * p_gradient
            # Average out the REINFORCE gradient.
            for g in gradient_U:
                g /= self._batch_size
            # Set the REINFORCE gradient for the optimizer.
            for index, p in enumerate(self.proposal.parameters()):
                p.grad = gradient_U[index].expand(p.size())
        # Apply an optimization step.
        self._optimizer_p.step()
        # Ensure the proposal is consistent.
        self.proposal.fix()

    def step(self, observables):
        # Draw a batch from the simulator
        inputs = self.proposal.sample((self._batch_size,))
        outputs = self.simulator(inputs)
        self._update_critic(observables, outputs)
        self._update_proposal(inputs, outputs)

    def infer(self, observables, steps=10000):
        self.call_event(self.events.fit_start)
        for step in range(steps):
            self.call_event(self.events.step_start, step=step)
            self.step(observables)
            self.call_event(self.events.step_complete, step=step)
        self.call_event(self.events.fit_complete)

        return self.proposal
