"""
Base modules for simulations.
"""

import torch

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal



class Simulator(torch.nn.Module):

    def __init__(self):
        super(Simulator, self).__init__()

    def forward(self, thetas):
        """
        Method should return thetas, x_thetas.
        """
        raise NotImplementedError

    def terminate(self):
        pass



class NormalSimulator(Simulator):

    def __init__(self):
        super(NormalSimulator, self).__init__()

    def forward(self, thetas):
        thetas = thetas.view(-1, 1)
        N = Normal(thetas, 1.)
        x_thetas = N.sample(torch.Size([thetas.size(0)]))
        x_thetas = x_thetas.view(-1, 1)

        return thetas, x_thetas



class MultivariateNormalSimulator(Simulator):

    def __init__(self, dim):
        super(MultivariateNormal).__init__()
        self.dim = dim
        self.sigma = torch.eye(self.dim)

    def forward(self, thetas):
        thetas = thetas.view(-1, self.dim)
        N = MultivariateNormal(thetas, self.sigma)
        x_thetas = N.sample(torch.Size([thetas.size(0)]))
        x_thetas = x_thetas.view(-1, self.dim)

        return thetas, x_thetas
