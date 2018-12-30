"""
Transition proposals.
"""

import numpy as np
import torch

from torch.distributions.multivariate_normal import MultivariateNormal as MultivariateNormalDistribution
from torch.distributions.normal import Normal as NormalDistribution
from torch.distributions.uniform import Uniform as UniformDistribution



class Transition:

    def log_prob(self, thetas_current, thetas_next):
        raise NotImplementedError

    def sample(self, thetas, samples=1):
        raise NotImplementedError

    def is_symmetric(self):
        raise NotImplementedError



class SymmetricTransition(Transition):

    def is_symmetric(self):
        return True



class Normal(SymmetricTransition):

    def __init__(self, sigma):
        super(Normal, self).__init__()
        self.sigma = sigma

    def log_prob(self, thetas_current, thetas_next):
        normal = NormalDistribution(thetas_current, self.sigma)

        return normal.log_prob(thetas_next)

    def sample(self, thetas, samples=1):
        with torch.no_grad():
            size = torch.Size([samples])
            next_thetas = NormalDistribution(thetas.squeeze(), self.sigma).sample(size)
            next_thetas = next_thetas.view(-1, samples)

        return next_thetas



class MultivariateNormal(SymmetricTransition):

    def __init__(self, sigma):
        super(MultivariateNormal, self).__init__()
        self.sigma = sigma.squeeze().detach()
        self.dimsionality = self.sigma.size(0)

    def log_prob(self, thetas_current, thetas_next):
        normal = MultivariateNormalDistribution(thetas_current, self.sigma)

        return normal.log_prob(thetas_next)

    def sample(self, thetas, samples=1):
        x = []

        with torch.no_grad():
            thetas = thetas.view(-1, self.dimsionality)
            samples = torch.Size([samples])
            for theta in thetas:
                out = MultivariateNormal(theta.view(-1), self.sigma).sample(samples).view(1, samples, self.dimsionality)
                x.append(out)
            x = torch.cat(x, dim=0).squeeze()

        return x



class Uniform(SymmetricTransition):

    def __init__(self, bound_min, bounds_max):
        super(Uniform, self).__init__()
        self.bound_min = torch.tensor(bound_min).float().view(1, -1)
        self.bound_max = torch.tensor(bound_max).float().view(1, -1)
        self.dimensionality = self.bound_min.size(1)
        self.distribution = UniformDistribution(self.bound_min, self.bound_max)

    def log_prob(self, thetas_current, thetas_next):
        return self.distribution.log_prob(thetas_next)

    def sample(self, thetas, samples=1):
        x = []

        with torch.no_grad():
            pass

        return x
