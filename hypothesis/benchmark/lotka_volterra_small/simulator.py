r"""Simulator definition of the Lotka Volterra population model.

"""

import numpy as np
import torch

from hypothesis.benchmark.lotka_volterra import Simulator as BaseLotkaVolterraBenchmarkSimulator
from .util import Prior


class LotkaVolterraBenchmarkSimulator(BaseLotkaVolterraBenchmarkSimulator):
    r"""Simulation model of the Lotka Volterra population model.

    Implemented as a Markov Jump Process. Based on the implementation
    originally provided by George.

    In contrast to the original implementation, this particular problem
    setting only focusses on the Predator parameters,
    while effictively marginalizing over the Prey paramters.
    """

    def __init__(self, predators=50, prey=100, duration=50, dt=0.025):
        super(LoktaVolterraBenchmarkSimulator, self).__init__(
            predators=predators,
            prey=prey,
            duration=duration,
            dt=dt)
        self._prey_prior = Prior()

    @torch.no_grad()
    def forward(self, inputs, **kwargs):
        samples = []

        inputs = inputs.view(-1, 2).exp()
        latents = self._prey_prior.sample((len(inputs),)).exp()
        inputs = torch.cat([inputs, latents], dim=1)
        for theta in inputs:
            samples.append(self._simulate(theta).unsqueeze(0))

        return torch.cat(samples, dim=0)
