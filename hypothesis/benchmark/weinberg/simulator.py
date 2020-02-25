r"""This is a simulation of high energy particle collisions $e^+e^- \to \mu^+ \mu^-$.
The angular distributions of the particles can be used to measure the Weinberg angle
in the standard model of particle physics. If you get a PhD in particle physics,
you may learn how to calculate these distributions and interpret those equations to
learn that an effective way to infer this parameter is to run your particle accelerator
with a beam energy just above or below half the $Z$ boson mass (i.e. the optimal $\phi$
is just above and below 45 GeV).

Adapted from https://github.com/cranmer/active_sciencing/blob/master/demo_weinberg.ipynb

Original implementation by Lucas Heinrich and Kyle Cranmer
"""

import hypothesis
import numpy as np
import torch

from hypothesis.simulation import Simulator as BaseSimulator



class WeinbergSimulator(BaseSimulator):

    MZ = int(90)
    GFNom = float(1)

    def __init__(self, default_beam_energy=45.0):
        super(WeinbergSimulator, self).__init__()
        self.default_beam_energy = default_beam_energy

    def _a_fb(self, sqrtshalf, gf):
        sqrts = sqrtshalf * 2.
        A_FB_EN = np.tanh((sqrts - self.MZ) / self.MZ * 10)
        A_FB_GF = gf / self.GFNom

        return 2 * A_FB_EN * A_FB_GF

    def _diffxsec(self, costheta, sqrtshalf, gf):
        norm = 2. * ((1. + 1. / 3.))

        return ((1 + costheta**2) + self._a_fb(sqrtshalf, gf) * costheta) / norm

    def simulate(self, theta, psi):
        # theta = gf
        # psi = sqrtshalf
        sample = None

        x = np.linspace(-1, 1, 10000)
        maxval = np.max(self._diffxsec(x, psi, theta))
        while sample is None:
            xprop = np.random.uniform(-1, 1)
            ycut = np.random.random()
            yprop = self._diffxsec(xprop, psi, theta) / maxval
            if yprop / maxval < ycut:
                continue
            sample = xprop

        return torch.tensor(sample).view(1, 1)

    @torch.no_grad()
    def forward(self, inputs, experimental_configurations=None):
        outputs = []

        n = len(inputs)
        for index in range(n):
            theta = inputs[index]
            if experimental_configurations is not None:
                psi = experimental_configurations[index]
                x = self.simulate(theta.item(), psi.item())
            else:
                x = self.simulate(theta.item(), self.default_beam_energy)
            outputs.append(x)
        outputs = torch.cat(outputs, dim=0)

        return outputs
