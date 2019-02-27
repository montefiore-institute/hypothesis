import numpy as np
import torch

from hypothesis.simulation import Simulator



def allocate_observations(theta):
    inputs = torch.tensor(theta).view(1, 4).float()
    simulator = LotkaVolterraSimulator()
    output = simulator(inputs)

    return output



class LotkaVolterraSimulator(Simulator):

    def __init__(self, x=100, y=100, t=99):
        super(LotkaVolterraSimulator, self).__init__()
        self.x = x
        self.y = y
        self.t = t

    def generate(self, alpha, beta, gamma, delta):
        x = self.x
        y = self.y
        x_list = [x]
        y_list = [y]

        for i in range(self.t):
            dx = alpha * x - beta * x * y
            dy = delta * x * y - gamma * y
            x += dx
            y += dy
            x_list.append(x)
            y_list.append(y)

        return torch.FloatTensor([x_list, y_list])

    def forward(self, inputs):
        samples = []

        with torch.no_grad():
            batch_size = inputs.size(0)
            alphas, betas, gammas, deltas = position.split(1, dim=1)
            for batch_index in range(batch_size):
                alpha = alphas[batch_index]
                beta = betas[batch_index]
                gamma = gammas[batch_index]
                delta = deltas[batch_index]
                samples.append(self.generate(alpha, beta, gamma, delta))
            samples = torch.cat(samples, dim=0).contiguous()

        return samples
