"""
Evolutionary Strategies.
"""

import numpy as np
import torch

from cag.inference import Method
from cag.inference.avo import r1
from cag.util import sample



class ClassifierEvolutionaryStrategy(Method):

    def __init__(self, simulator,
                 classifier,
                 num_particles=32,
                 batch_size=32,
                 classifier_steps=2,
                 lr_classifier=0.0001,
                 k=5):
        super(ClassifierEvolutionaryStrategy, self).__init__(simulator)
        self.classifier = classifier
        self.num_particles = num_particles
        self.batch_size = batch_size
        self.particles = None
        self.k = k
        self._classifier_steps = classifier_steps
        self._lr_classifier = lr_classifier
        self._criterion = torch.nn.BCELoss()
        self._real = torch.zeros(self.batch_size, 1)
        self._fake = torch.ones(self.batch_size, 1)
        self._o_classifier = None
        self._thetas_simulated = None
        self._x_thetas_simulated = None

    def _reset(self):
        self._num_simulations = 0
        self._allocate_optimizer()
        self._thetas_simulated = None
        self._x_thetas_simulated = None
        self.particles = None

    def _allocate_optimizer(self):
        self._o_classifier = None
        self._o_classifier = torch.optim.Adam(
            self.classifier.parameters(), lr=self._lr_classifier
        )

    def _add_simulated(self, thetas, x_thetas):
        with torch.no_grad():
            t = self._thetas_simulated
            x_t = self._x_thetas_simulated
            if t is None:
                self._thetas_simulated = thetas.clone()
                self._x_thetas_simulated = x_thetas.clone()
            else:
                self._thetas_simulated = torch.cat([t, thetas], dim=0)
                self._x_thetas_simulated = torch.cat([x_t, x_thetas], dim=0)

    def _sample_simulated(self):
        thetas = sample(self._thetas_simulated, self.batch_size)
        x_thetas = sample(self._x_thetas_simulated, self.batch_size)

        return thetas, x_thetas

    def _populate(self, initializer):
        with torch.no_grad():
            size = torch.Size([self.num_particles])
            thetas = initializer.sample(size)
            self.particles = thetas

    def _optimize_classifier(self, x_o):
        _, x_fake = self._sample_simulated()
        x_real = sample(x_o, self.batch_size)
        x_real.requires_grad = True
        y_real = self.classifier(x_real)
        y_fake = self.classifier(x_fake)
        critic = self.classifier
        loss = -(critic(x_real).log() + (1 - critic(x_fake)).log()).mean()
        gradients = torch.autograd.grad(loss, critic.parameters(), create_graph=True)
        gradient_norm = 0
        for gradient in gradients:
           gradient_norm = gradient_norm + (gradient ** 2).norm(p=1)
        gradient_norm /= len(gradients)
        loss = loss + gradient_norm
        self._o_classifier.zero_grad()
        loss.backward()
        self._o_classifier.step()
        x_real.requires_grad = False

    def _update(self, x_o, x_thetas):
        pi = self.classifier(x_thetas)
        top, top_indices = torch.topk(pi, self.k, dim=0, largest=True)
        particles_hat = self.particles[top_indices].squeeze()
        mean = particles_hat.mean(dim=0)
        sigma = particles_hat.std(dim=0)
        pi_hat = pi[top_indices].squeeze()
        sigma_correction = pi.mean() * torch.ones(self.particles.size(1)) * (self.classifier(sample(x_o, self.k)).mean() - pi_hat.mean())
        sigma += sigma_correction
        new_particles = (torch.randn(self.num_particles - self.k, self.particles.size(1)) * sigma) + mean
        self.particles = torch.cat([new_particles, particles_hat], dim=0).detach()

    def step(self, x_o):
        thetas, x_thetas = self.simulator(self.particles)
        self.particles = thetas
        self._add_simulated(thetas, x_thetas)
        for training_iteration in range(self._classifier_steps):
            self._optimize_classifier(x_o)
        self._update(x_o, x_thetas)

    def infer(self, x_o, initializer, num_steps=1000):
        self._reset()
        self._populate(initializer)

        for step in range(num_steps):
            self.step(x_o)
        with torch.no_grad():
            mean = self.particles.mean(dim=0)
            std = self.particles.std(dim=0)

        return mean, std
