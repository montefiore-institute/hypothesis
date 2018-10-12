"""
Elastic Particle Filtering
"""

import numpy as np
import torch
import random

from cag.inference import Method



class ElasticParticleFiltering(Method):

    def __init__(self, simulator,
                 regressor,
                 num_particles=32,
                 elasticity=.01):
        super(ElasticParticleFiltering, self).__init__(simulator)
        self.regressor = regressor
        self.num_particles = num_particles
        self.particles = None
        self._criterion = torch.nn.MSELoss()
        self._elasticity = elasticity
        self._o_regressor = None
        self._num_simulations = 0
        self._D = []

    def _reset(self):
        self._D = []
        self.particles = None
        self._num_simulations = 0
        self._allocate_optimizers()

    def _populate(self, initializer):
        with torch.no_grad():
            size = torch.Size([self.num_particles])
            thetas = initializer.sample(size)
            self.particles = thetas

    def _allocate_optimizers(self):
        self._o_regressor = torch.optim.Adam(self.regressor.parameters())

    def _optimize_regressor(self):
        thetas, x_thetas = self._D[-1]
        thetas_hat = self.regressor(x_thetas)
        loss = self._criterion(thetas_hat, thetas)
        gradients = torch.autograd.grad(loss, self.regressor.parameters(), create_graph=True)
        gradient_norm = 0
        for gradient in gradients:
            gradient_norm = gradient_norm + (gradient ** 2).norm(p=1)
        gradient_norm /= len(gradients)
        loss = loss + gradient_norm
        self._o_regressor.zero_grad()
        loss.backward()
        self._o_regressor.step()

    def step(self, x_o):
        # Sample from the simulator.
        self.particles, x_thetas = self.simulator(self.particles)
        self._D.append((self.particles, x_thetas))
        # Update regression model.
        self._optimize_regressor()
        # Update particles.
        attractor = self.regressor(x_o).mean(dim=0)
        with torch.no_grad():
            self.particles += self._elasticity * (attractor - self.particles)
        self.particles = self.particles.detach()

    def infer(self, x_o, initializer, num_steps=1000):
        self._reset()
        self._populate(initializer)

        for step in range(num_steps):
            self.step(x_o)
        with torch.no_grad():
            mean = self.particles.mean(dim=0)
            std = self.particles.std(dim=0)

        return mean, std
