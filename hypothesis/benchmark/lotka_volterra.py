import numpy as np
import torch
import hypothesis.benchmark.markov_jump_processes as mjp

from hypothesis.simulation import Simulator




def allocate_observations(theta=None, num_observations=1):
    # Check if a theta has been specified.
    if theta is None:
        inputs = torch.tensor([0.01, .5, 1, 0.01]).float()
    else:
        inputs = torch.tensor(theta).float()
    inputs = inputs.view(-1, 4).repeat(num_observations)
    simulator = LotkaVolterraSimulator()
    outputs = simulator(inputs)

    return outputs



class LotkaVolterraSimulator(Simulator):

    def __init__(self, prey=100, predator=50, t=30, dt=0.2):
        super(LotkaVolterraSimulator, self).__init__()
        self.duration = t
        self.initial_predator = predator
        self.initial_prey = prey
        self.dt = dt
        self.lv = None
        self._initialize()

    def _initialize(self):
        self.lv = mjp.LotkaVolterra([self.initial_prey, self.initial_predator], None)

    def _generate(self, theta):
        self._initialize()
        self.lv.reset([self.initial_prey, self.initial_predator], theta.numpy())
        states = self.lv.sim_time(self.dt, self.duration, max_n_steps=10000, rng=np.random)
        return torch.tensor(states.flatten()).float().view(1, -1, 2)

    def forward(self, thetas):
        outputs = []

        thetas = thetas.view(-1, 4)
        for theta in thetas:
            theta = theta.view(-1)
            x_theta = self._generate(theta)
            outputs.append(x_theta)
        outputs = torch.cat(outputs, dim=0)

        return outputs
