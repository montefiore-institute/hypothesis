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



class PythiaSimulator(Simulator):

    def __init__(self, workers=4, options=None, detector=None):
        if not options:
            options = default_options()
        if not detector:
            detector = default_detector()
        self._mill = pm.ParameterizedPythiaMill(
            detector, options, batch_size=1, n_workers=workers)

    def forward(self, thetas):
        raise NotImplementedError

    def terminate(self):
        if self._mill:
            self._mill.terminate()
            self._mill = None
