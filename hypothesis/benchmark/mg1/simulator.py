r"""Simulation model of the M/G/1 queuing model.

This model describes a queuing system of continuously arriving jobs by a
single server. The time it takes to process every job is uniformly
distributed in the interval :math:`[\theta_1, \theta_2]`. The arrival
between two consecutive jobs is exponentially distributed according to
the rate :math:`\theta_3`. That is, for
every job :math:`i` we have the processing time :math:`p_i` , an arrival
time :math:`a_i` and the time :math:`l_i` at which the job left the queue.

"""

import numpy as np
import numpy.random as rng
import torch

from hypothesis.simulation import BaseSimulator


class Simulator(BaseSimulator):

    def __init__(self, percentiles=5, steps=50):
        super(Simulator, self).__init__()
        self.num_percentiles = int(percentiles)
        self.num_steps = int(steps)

    @torch.no_grad()
    def _generate(self, input):
        input = input.view(-1)
        p1 = input[0].item()
        p2 = input[1].item()
        p3 = input[2].item()
        # Service / processing time.
        sts = (p2 - p1) * rng.random(self.num_steps) + p1
        # Interarrival times.
        iats = -np.log(1.0 - rng.rand(self.num_steps)) / p3
        # Arrival times.
        ats = np.cumsum(iats)
        # Interdeparture and departure times.
        idts = np.empty(self.num_steps)
        dts = np.empty(self.num_steps)
        idts[0] = sts[0] + ats[0]
        dts[0] = idts[0]
        for i in range(1, self.num_steps):
            idts[i] = sts[i] + max(0.0, ats[i] - dts[i-1])
            dts[i] = dts[i-1] + idts[i]
        # Compute the observation.
        perc = np.linspace(0.0, 100.0, self.num_percentiles)
        stats = np.percentile(idts, perc)

        return torch.tensor(stats).float().view(1, -1)

    @torch.no_grad()
    def forward(self, inputs):
        samples = []

        inputs = inputs.view(-1, 3)
        for input in inputs:
            x_out = self._generate(input)
            samples.append(x_out.view(1, -1))

        return torch.cat(samples, dim=0)
