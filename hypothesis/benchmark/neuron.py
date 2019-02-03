import hypothesis
import hypothesis.simulation
import numpy as np
import torch
import neuron

from neuron import h



def initialize_simulator():
    # Create the neuron.
    h.load_file('stdrun.hoc')
    h.load_file('sPY_template')
    h('objref IN')
    h('IN = new sPY()')
    h.celsius = 36
    # Create electrode.
    h('objref El')
    h('IN.soma El = new IClamp(0.5)')
    h('El.del = 0')
    h('El.dur = 100')
    # Set simulation time and initial milli-voltage.
    h.tstop = 100.0
    h.v_init = -70.0
    # Record voltage
    h('objref v_vec')
    h('objref t_vec')
    h('t_vec = new Vector()')
    h('t_vec.indgen(0, tstop, dt)')
    h('v_vec = new Vector()')
    h('IN.soma v_vec.record(&v(0.5), t_vec)')


def simulator(inputs):
    outputs = []

    # Hudgekey model has 12 parameters to be inferred.
    inputs = inputs.view(-1, 12)
    for input in inputs:
        input = input.view(-1)
        # Configure the simulation.
        h.IN.soma[0](0.5).g_pas       = input[0] # g_leak
        h.IN.soma[0](0.5).gnabar_hh2  = input[1] # gbar_Na
        h.IN.soma[0](0.5).gkbar_hh2   = input[2] # gbar_K
        h.IN.soma[0](0.5).gkbar_im    = input[3] # gbar_M
        h.IN.soma[0](0.5).e_pas       = input[4] # E_leak
        h.IN.soma[0](0.5).ena         = input[5] # E_Na
        h.IN.soma[0](0.5).ek          = input[6] # E_K
        h.IN.soma[0](0.5).vtraub_hh2  = input[7] # V_T
        h.IN.soma[0](0.5).kbetan1_hh2 = input[8] # k_betan1
        h.IN.soma[0](0.5).kbetan2_hh2 = input[9] # k_betan2
        h.taumax_im                   = input[10] # tau_max
        sigma                         = input[11] # sigma
        # Set up current injection of noise
        Iinj = rng.normal(0.5, sigma, np.array(h.t_vec).size)
        Iinj_vec = h.Vector(Iinj)
        Iinj_vec.play(h.El._ref_amp, h.t_vec)
        # Initialize and run
        neuron.init()
        h.finitialize(h.v_init)
        neuron.run(h.tstop)
        outputs.append(torch.tensor(h.v_vec))
    outputs = torch.cat(outputs, dim=0).view(-1, 12)

    return outputs


def allocate_observations(theta, observations=10000):
    simulator = Simulator()


class Simulator(hypothesis.simulation.Simulator):

    def __init__(self):
        super(Simulator, self).__init__()
        initialize_simulator()

    def forward(self, inputs):
        return simulator(inputs)
