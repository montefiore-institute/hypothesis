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
                 classifier_steps=2,
                 lr_classifier=0.0001,
                 r1_regularization=10.):
        super(ClassifierEvolutionaryStrategy, self).__init__(simulator)
        self.classifier = classifier
        self.num_particles = num_particles
        self.particles = None
        self._classifier_steps = classifier_steps
        self._lr_classifier = _lr_classifier
        self._r1 = float(r1_regularization)
        self._criterion = torch.nn.BCELoss()
        self._real = torch.zeros(self.num_particles, 1)
        self._fake = torch.ones(self.num_particles, 1)
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
        thetas = sample(self._thetas_simulated, self.num_particles)
        x_thetas = sample(self._x_thetas_simulated, self.num_particles)

        return thetas, x_thetas

    def _populate(self, initializer):
        with torch.no_grad():
            size = torch.Size([self.num_particles])
            thetas = initializer.sample(size)
            self.particles = thetas

    def _optimize_classifier(self, x_o):
        _, x_fake = self._sample_simulated()
        x_real = sample(x_o)
        x_real.requires_grad = True
        y_real = self.classifier(x_real)
        y_fake = self.classifier(y_fake)
        loss = (self._criterion(y_real, self._real) + self._criterion(y_fake, self._fake)) / 2.
        loss = loss + self._r1 * r1(y_real, x_real).mean()
        self._o_classifier.zero_grad()
        loss.backward()
        self._o_classifier.step()
        x_real.requires_grad = False

    def _update(self, k=5):
        pi = self.classifier(self.particles)
        top, top_indices = torch.topk(pi, k, dim=0, largest=True)
        particles_hat = self.particles[top_indices]
        mean = particles_hat.mean(dim=0)
        sigma = particles_hat.std(dim=0)
        new_particles = ((torch.randn(self.num_particles - k, self.particles.size(0)) * sigma) + mean).squeeze()
        self.particles = torch.cat([new_particles, particles_hat], dim=0).detach()
        print(self.particles.mean(dim=0))

    def step(self, x_o):
        thetas, x_thetas = self.simulator(self.particles)
        self._add_simulated(thetas, x_thetas)
        for training_iteration in range(self._classifier_steps):
            self._optimize_classifier(x_o)
        self._update()

    def infer(self, x_o, initializer, num_steps=1000):
        self._reset()
        self._populate(initializer)

        for step in range(num_steps):
            self.step(x_o)
        with torch.no_grad():
            mean = self.particles.mean(dim=0)
            std = self.particles.std(dim=0)

        return mean, std
