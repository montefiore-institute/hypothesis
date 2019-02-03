"""
Weinberg bencharmking problem.

Thanks to Lukas Heinrich for providing the Weinberg simulator.
http://www.lukasheinrich.com/
Twitter: @lukasheinrich_
Github: @lukasheinrich
"""

import numpy as np
import torch

from hypothesis.simulation import Simulator



def simulator(inputs):
    with torch.no_grad():
        # Ensure the model parameters have the correct shape.
        inputs = inputs.view(-1, 2)
        outputs = []
        for theta in inputs:
            e_beam = (theta[0].item() - 40.) / (50. - 40.)
            g_f = (theta[1].item() - .5 ) / (1.5 - .5)
            output = torch.tensor(weinberg_rej_sample_costheta(np.array([e_beam, g_f]), 1))
            outputs.append(output)
        outputs = torch.cat(outputs, dim=0).float().view(-1, 1)

    return outputs


def allocate_observations(theta, observations=10000):
    inputs = torch.tensor(theta).float().repeat(observations).view(-1, 2)
    outputs = simulator(inputs)

    return outputs


class WeinbergSimulator(Simulator):

    def __init__(self):
        super(WeinbergSimulator, self).__init__()

    def forward(self, inputs):
        return simulator(inputs)


def weinberg_a_fb(sqrtshalf, gf):
    MZ = 90
    GFNom = 1.0
    sqrts = sqrtshalf * 2.
    A_FB_EN = np.tanh((sqrts - MZ) / MZ * 10)
    A_FB_GF = gf / GFNom

    return 2 * A_FB_EN * A_FB_GF


def weinberg_diffxsec(costheta, sqrtshalf, gf):
    norm = 2. * (1. + 1. / 3.)
    return ((1 + costheta ** 2) + weinberg_a_fb(sqrtshalf, gf) * costheta) / norm


def weinberg_rej_sample_costheta(theta, n_samples):
    sqrtshalf = theta[0] * (50-40) + 40
    gf = theta[1] * (1.5 - 0.5) + 0.5

    ntrials = 0
    samples = []
    x = np.linspace(-1, 1, num=1000)
    maxval = np.max(weinberg_diffxsec(x, sqrtshalf, gf))

    while len(samples) < n_samples:
        ntrials = ntrials+1
        xprop = np.random.uniform(-1, 1)
        ycut = np.random.random()
        yprop = weinberg_diffxsec(xprop, sqrtshalf, gf)
        if yprop/maxval < ycut:
            continue
        samples.append(xprop)

    return np.array(samples)
