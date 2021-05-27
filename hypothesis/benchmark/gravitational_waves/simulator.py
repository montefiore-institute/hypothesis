r"""Simulator definition of the Gravitational Wave benchmark problem.

This specific model marginalizes over the mass parameters of the black holes.
The problem dimensionality of the inputs therefore reduces to 2.

Inspired by https://github.com/timothygebhard/ggwd
"""

import numpy as np
import os
import pycbc
import torch

from hypothesis.simulation import BaseSimulator
from lal import LIGOTimeGPS
from pycbc.detector import Detector
from pycbc.distributions import JointDistribution
from pycbc.distributions import read_constraints_from_config
from pycbc.distributions import read_distributions_from_config
from pycbc.distributions import read_params_from_config
from pycbc.noise import noise_from_psd
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.waveform import get_td_waveform
from pycbc.workflow import WorkflowConfigParser



class GravitationalWaveBenchmarkSimulator(BaseSimulator):
    r"""Simulation model associated with the gravitational waves benchmark.

    Marginalizes over the mass parameters. The dimensionality of the
    problem is therefore reduces to 2.
    """

    def __init__(self, config_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_file.ini")):
        super(GravitationalWaveBenchmarkSimulator, self).__init__()

        workflow_config_parser = WorkflowConfigParser(configFiles=[config_file])
        self.variable_arguments, self.static_arguments = read_params_from_config(workflow_config_parser)
        dist = read_distributions_from_config(workflow_config_parser)
        self.pval = JointDistribution(self.variable_arguments, *dist)

    def _simulate_gw(self, mass1, mass2):
        param_values = self.pval.rvs()[0]
        params = dict(zip(self.variable_arguments, param_values))
        params["mass1"] = mass1
        params["mass2"] = mass2

        td_length = int(self.static_arguments['waveform_length'] * self.static_arguments["sampling_rate"])
        delta_t = 1./self.static_arguments["sampling_rate"]
        fd_length = int(td_length / 2.0 + 1)
        delta_f = 1./self.static_arguments["waveform_length"]
        event_time = self.static_arguments["seconds_before_event"]

        h_plus, h_cross = get_td_waveform(approximant=self.static_arguments["approximant"],
                                          delta_t=delta_t,
                                          delta_f=delta_f,
                                          f_lower=self.static_arguments["f_lower"],
                                          coa_phase=params["coa_phase"],
                                          distance=params["distance"],
                                          inclination=params["inclination"],
                                          mass1=params["mass1"],
                                          mass2=params["mass2"],
                                          spin1z=params["spin1z"],
                                          spin2z=params["spin2z"])

        h_plus.resize(td_length)
        h_cross.resize(td_length)

        detectors = {'H1': Detector('H1'), 'L1': Detector('L1')}
        signals = {}

        for detector_name in ('H1', 'L1'):

            detector = detectors[detector_name]

            f_plus, f_cross = detector.antenna_pattern(right_ascension=params["ra"],
                                                       declination=params["dec"],
                                                       polarization=params["polarization"],
                                                       t_gps=100)

            delta_t_h1 = detector.time_delay_from_detector(other_detector=detectors['H1'],
                                                           right_ascension=params["ra"],
                                                           declination=params["dec"],
                                                           t_gps=100)

            signal = f_plus * h_plus + f_cross * h_cross

            offset = 100 + delta_t_h1 + signal.start_time
            signal = signal.cyclic_time_shift(offset)
            signal.start_time = event_time - 100

            signals[detector_name] = signal


        psd = aLIGOZeroDetHighPower(length=fd_length, delta_f=delta_f, low_freq_cutoff=self.static_arguments["f_lower"])
        noise_length = int(self.static_arguments["noise_interval_width"] * self.static_arguments["sampling_rate"])
        start_time = event_time - self.static_arguments["noise_interval_width"] / 2

        noise = {}
        for det in ("H1", "L1"):
            noise[det] = noise_from_psd(length=noise_length, delta_t=delta_t, psd=psd)
            noise[det]._epoch = LIGOTimeGPS(start_time)

        strain = {}

        for det in ("H1", "L1"):
            strain[det] = noise[det].add_into(signals[det])

        for det in ("H1", "L1"):
            strain[det] = strain[det].whiten(segment_duration=self.static_arguments["whitening_segment_duration"],
                                             max_filter_duration=self.static_arguments["whitening_max_filter_duration"],
                                             remove_corrupted=False)

            strain[det] = strain[det].highpass_fir(frequency=self.static_arguments["bandpass_lower"],
                                                   remove_corrupted=False, order=512)

        a = event_time - self.static_arguments["seconds_before_event"]
        b = event_time + self.static_arguments["seconds_after_event"]

        for det in ("H1", "L1"):
            strain[det] = strain[det].time_slice(a, b)

        return strain["H1"], strain["L1"]


    @torch.no_grad()
    def forward(self, inputs, **kwargs):
        samples = []

        inputs = inputs.view(-1, 2)
        for input in inputs:
            h1, l1 = self._simulate_gw(input[0], input[1])
            h1 = torch.tensor(h1)
            l1 = torch.tensor(l1)
            x_out = torch.stack([h1, l1], 0)
            samples.append(x_out)

        return torch.stack(samples, dim=0)
