import hypothesis
import numpy as np
import torch

from hypothesis.simulation import Simulator as BaseSimulator
from torch.distributions.binomial import Binomial



class DeathModelSimulator(BaseSimulator):

    def __init__(self, population_size=1000, default_measurement_time=1.0, step_size=0.01):
        super(DeathModelSimulator, self).__init__()
        self.default_measurement_time = torch.tensor(default_measurement_time).float()
        self.population_size = int(population_size)
        self.step_size = float(step_size)

    def simulate(self, theta, psi):
        # theta = [beta, gamma]
        # psi = tau
        # sample = [S(tau), I(tau), R(tau)]
        infection_rate = theta.item()
        design = psi.item()
        I = 0
        t = 0.0
        n_steps = int(psi / self.step_size / 10)
        deltas = []
        for _ in range(n_steps):
            S = self.population_size - I
            if S == 0:
                break
            p_inf = 1 - np.exp(-infection_rate * t)
            delta_I = int(Binomial(S, p_inf).sample())
            I += delta_I
            deltas.append(delta_I)
            t += self.step_size

        return torch.tensor(I).float()

    @torch.no_grad()
    def forward(self, inputs, experimental_configurations=None):
        outputs = []

        n = len(inputs)
        for index in range(n):
            theta = inputs[index]
            if experimental_configurations is not None:
                psi = experimental_configurations[index]
                x = self.simulate(theta, psi)
            else:
                x = self.simulate(theta, self.default_measurement_time)
            outputs.append(x.view(1, -1))
        outputs = torch.cat(outputs, dim=0)

        return outputs
