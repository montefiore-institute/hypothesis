import hypothesis
import numpy as np
import torch

from hypothesis.simulation import Simulator as BaseSimulator
from torch.distributions.binomial import Binomial



class SIRSimulator(BaseSimulator):

    def __init__(self, population_size=1000, default_measurement_time=1.0, step_size=0.01):
        super(SIRSimulator, self).__init__()
        self.default_measurement_time = torch.tensor(default_measurement_time).float()
        self.population_size = int(population_size)
        self.step_size = float(step_size)

    def simulate(self, theta, psi):
        # theta = [beta, gamma]
        # psi = tau
        # sample = [S(tau), I(tau), R(tau)]
        beta = theta[0].item()
        gamma = theta[1].item()
        psi = psi.item()
        S = self.population_size - 1
        I = 1
        R = 0
        n_steps = int(psi / self.step_size)
        for i in range(n_steps):
            if I == 0: # State will remain the same.
                break
            delta_I = int(Binomial(S, beta * I / self.population_size).sample())
            delta_R = int(Binomial(I, gamma).sample())
            S -= delta_I
            I = I + delta_I - delta_R
            R += delta_R

        return torch.tensor([S, I, R]).float()

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
