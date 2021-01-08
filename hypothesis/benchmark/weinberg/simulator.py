import hypothesis as h
import numpy as np
import torch

from hypothesis.simulation import BaseSimulator


class Simulator(BaseSimulator):
    r"""This is a simulation of high energy particle collisions $e^+e^- \to \mu^+ \mu^-$.
    The angular distributions of the particles can be used to measure the Weinberg angle
    in the standard model of particle physics. If you get a PhD in particle physics,
    you may learn how to calculate these distributions and interpret those equations to
    learn that an effective way to infer this parameter is to run your particle accelerator
    with a beam energy just above or below half the $Z$ boson mass (i.e. the optimal $\phi$
    is just above and below 45 GeV).

    Adapted from https://github.com/cranmer/active_sciencing/blob/master/demo_weinberg.ipynb

    Original implementation by Lucas Heinrich and Kyle Cranmer

    .. code-block:: python

        from hypothesis.benchmark.weinberg import Prior
        from hypothesis.benchmark.weinberg import Simulator

        prior = Prior()
        simulator = Simulator()

        inputs = prior.sample((10,))  # Draw 10 samples from the prior
        outputs = simulator(inputs)

        # You can also batch with respect to the experimental configurations
        from hypothesis.benchmark.weinberg import PriorExperiment

        prior_experiment = PriorExperiment()
        beam_energies = prior_experiment.sample((10,))
        outputs = simulator(inputs, beam_energies)
    """

    MZ = int(90)
    GFNom = float(1)

    def __init__(self, default_beam_energy=40.0, num_samples=1):
        super(Simulator, self).__init__()
        self._num_samples = int(num_samples)
        self._default_beam_energy = float(default_beam_energy)

    def _a_fb(self, sqrtshalf, gf):
        sqrts = sqrtshalf * 2.
        A_FB_EN = np.tanh((sqrts - self.MZ) / self.MZ * 10)
        A_FB_GF = gf / self.GFNom

        return 2 * A_FB_EN * A_FB_GF

    def _diffxsec(self, costheta, sqrtshalf, gf):
        norm = 2. * ((1. + 1. / 3.))

        return ((1 + costheta**2) + self._a_fb(sqrtshalf, gf) * costheta) / norm

    @torch.no_grad()
    def _simulate(self, theta, psi):
        # theta = gf
        # psi = sqrtshalf
        samples = []

        for _ in range(self._num_samples):
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
            sample = torch.tensor(sample).view(1, 1)
            samples.append(sample)

        return torch.cat(samples, dim=1)

    @torch.no_grad()
    def forward(self, inputs, experimental_configurations=None):
        r"""Executes the forward pass of the simulation model.

        :param inputs: Free parameters (the Fermi constant).
        :param experimental_configurations: Optional experimental
                                            parameters describing the beam energy.

        .. note::

            This method accepts a batch of corresponding inputs and optional
            experimental configuration pairs.
        """
        outputs = []

        inputs = inputs.view(-1, 1)
        n = len(inputs)
        if experimental_configurations is not None:
            experimental_configurations = experimental_configurations.view(-1, 1)
        else:
            experimental_configurations = n * [self._default_beam_energy]
            experimental_configurations = torch.tensor(experimental_configurations)
        for index in range(n):
            theta = inputs[index]
            psi = experimental_configurations[index]
            x = self.simulate(theta.item(), psi.item())
            outputs.append(x)
        outputs = torch.cat(outputs, dim=0)

        return outputs
