"""
Pythia benchmark.

This requires a working installation of Pythia 8 and Pythiamill.
"""

import torch
import numpy as np
import pythiamill as pm

from hypothesis.simulation import Simulator



def default_options():
    pythia_options = [
        'Print:quiet = on',
        'Init:showProcesses = off',
        'Init:showMultipartonInteractions = off',
        'Init:showChangedSettings = off',
        'Init:showChangedParticleData = off',
        'Next:numberCount=0',
        'Next:numberShowInfo=0',
        'Next:numberShowEvent=0',
        'Stat:showProcessLevel=off',
        'Stat:showErrors=on',

        "Beams:idA = 11",
        "Beams:idB = -11",
        "Beams:eCM = 91.2",
        "WeakSingleBoson:ffbar2gmZ = on",
        "23:onMode = off",
        "23:onIfMatch = 1 -1",
        "23:onIfMatch = 2 -2",
        "23:onIfMatch = 3 -3",
        "23:onIfMatch = 4 -4",
        "23:onIfMatch = 5 -5",

        "Tune:ee = 7"]

    return pythia_options


def default_detector(resolution=32):
    detector = pm.utils.SphericalTracker(
        is_binary=False,
        max_pseudorapidity=5.0,
        pseudorapidity_steps=resolution, phi_steps=resolution,
        n_layers=1, R_min=10.0, R_max=10.0)

    return detector


def allocate_observations(theta, num_observations=100000, resolution=32):
    theta_true = torch.tensor([float(theta)]).float()
    simulator = PythiaDetectorOffsetSimulator(resolution=resolution)
    _, x_o = simulator(torch.cat([theta_true] * num_observations, dim=0))
    simulator.terminate()

    return theta_true, x_o



class PythiaDetectorOffsetSimulator(Simulator):

    def __init__(self, workers=4, options=None, detector=None, resolution=32):
        super(PythiaDetectorOffsetSimulator, self).__init__()
        if not options:
            options = default_options()
        if not detector:
            detector = default_detector(resolution)
        self._mill = pm.ParametrizedPythiaMill(
            detector, options, batch_size=1, n_workers=workers)
        self._resolution = resolution

    def forward(self, thetas):
        parameters = []
        x_thetas = []

        # Submit the parameter requests to PythiaMill.
        for theta in thetas:
            theta = theta.item()
            self._mill.request(theta)
        # Retrieve the simulated observations from the simulator.
        for i in range(thetas.size(0)):
            theta, x_theta = self._mill.retrieve()
            parameters.append(theta)
            x_thetas.append(x_theta)
        thetas = torch.tensor(parameters).view(-1, 1)
        x_thetas = np.array(x_thetas).reshape(-1, self._resolution ** 2)
        x_thetas = torch.tensor(x_thetas)
        x_thetas = x_thetas.view(-1, self._resolution, self._resolution).log1p()

        return thetas, x_thetas

    def terminate(self):
        if self._mill:
            self._mill.terminate()
            self._mill = None
