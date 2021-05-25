r"""Simulator definition of the Lotka Volterra population model.

"""

import numpy as np
import torch

from hypothesis.exception import SimulationTimeError
from hypothesis.simulation import BaseSimulator


class LotkaVolterraBenchmarkSimulator(BaseSimulator):
    r"""Simulation model of the Lotka Volterra population model.

    Implemented as a Markov Jump Process. Based on the implementation
    originally provided by George.
    """

    def __init__(self, predators=50, prey=100, duration=50, dt=0.025):
        super(BaseSimulator, self).__init__()
        self._initial_state = np.array([predators, prey])
        self._duration = float(duration)
        self._dt = float(dt)

    @torch.no_grad()
    def _simulate(self, theta):
        theta = theta.view(-1).numpy()
        steps = int(self._duration / self._dt) + 1
        states = np.zeros((steps, 2))
        state = np.copy(self._initial_state)
        for step in range(steps):
            x, y = state
            xy = x * y
            propensities = np.array([xy, x, y, xy])
            rates = theta * propensities
            total_rate = sum(rates)
            normalized_rates = rates / total_rate
            if total_rate <= 0.00001:
                break
            transition = np.random.choice([0, 1, 2, 3], p=normalized_rates)
            if transition == 0:
                state[0] += 1  # Increase predator population by 1
            elif transition == 1:
                state[0] -= 1  # Decrease predator population by 1
            elif transition == 2:
                state[1] += 1  # Increase prey population by 1
            else:
                state[1] -= 1  # Decrease prey population by 1
            states[step, :] = np.copy(state)

        return torch.from_numpy(states)

    @torch.no_grad()
    def forward(self, inputs, **kwargs):
        samples = []

        inputs = inputs.view(-1, 4).exp()
        for theta in inputs:
            samples.append(self._simulate(theta).unsqueeze(0))

        return torch.cat(samples, dim=0)
