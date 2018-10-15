"""
Weinberg benchmarking problem.

Thanks to Lukas Heinrich for providing the Weinberg simulator.
http://www.lukasheinrich.com/
Twitter: @lukasheinrich_
Github: @lukasheinrich
"""

import numpy as np
import torch

from cag.simulation import Simulator



def simulator(thetas):
    with torch.no_grad():
        thetas = thetas.view(-1, 2)
        samples = []
        for theta in thetas:
            e_beam = (theta[0].item() - 40.) / (50. - 40.)
            g_f = (theta[1].item() - .5 ) / (1.5 - .5)
            x_theta = torch.tensor(weinberg_rej_sample_costheta(np.array([e_beam, g_f]), 1))
            samples.append(x_theta)
        samples = torch.cat(samples, dim=0).float().view(-1, 1)

    return thetas, samples


def allocate_observations(theta, num_observations=100000):
    theta = torch.tensor(theta).view(1, 2)
    thetas = torch.cat([theta] * num_observations, dim=0).view(-1, 2)
    _, x_o = simulator(thetas)

    return theta, x_o


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


class WeinbergSimulator(Simulator):

    def __init__(self):
        super(WeinbergSimulator, self).__init__()

    def forward(self, thetas):
        return simulator(thetas)

    def terminate(self):
        pass
