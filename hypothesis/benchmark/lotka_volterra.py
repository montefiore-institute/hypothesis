import numpy as np
import torch
from scipy import integrate

from hypothesis.simulation import Simulator



def allocate_observations(theta):
    inputs = torch.tensor(theta).view(-1, 4).float()
    simulator = LotkaVolterraSimulator()
    output = simulator(inputs)

    return output



class LotkaVolterraSimulator(Simulator):

    def __init__(self, x=10, y=10, t=100, resolution=0.001):
        super(LotkaVolterraSimulator, self).__init__()
        self.x = x
        self.y = y
        self.t = t
        self.resolution = resolution

    def generate(self, alpha, beta, gamma, delta):

        def dX_dt(X, t=0):
            return np.array([alpha * X[0] - beta * X[0] * X[1], delta * X[0] * X[1] - gamma * X[1]])

        t = np.linspace(0, self.t, int(self.t/self.resolution))
        X0 = np.array([self.x, self.y])
        X = integrate.odeint(dX_dt, X0, t)
        return torch.FloatTensor(X)

    def forward(self, inputs):
        samples = []

        with torch.no_grad():
            batch_size = inputs.size(0)
            alphas, betas, gammas, deltas = inputs.split(1, dim=1)
            for batch_index in range(batch_size):
                alpha = alphas[batch_index]
                beta = betas[batch_index]
                gamma = gammas[batch_index]
                delta = deltas[batch_index]
                samples.append(self.generate(alpha, beta, gamma, delta))
            samples = torch.cat(samples, dim=0).contiguous()

        return samples
