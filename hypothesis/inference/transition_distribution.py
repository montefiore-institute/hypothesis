r""""""

import numpy as np
import torch

from torch.distributions.multivariate_normal import MultivariateNormal as MultivariateNormalDistribution
from torch.distributions.normal import Normal as NormalDistribution
from torch.distributions.uniform import Uniform as UniformDistribution



class Transition:
    r""""""

    def log_prob(self, xs, conditionals):
        raise NotImplementedError

    def sample(self, xs, samples=1):
        raise NotImplementedError

    def is_symmetrical(self):
        raise NotImplementedError



class SymmetricalTransition(Transition):

    def is_symmetrical(self):
        return True



class AsymmetricalTransition(Transition):

    def is_symmetrical(self):
        return False



class Normal(SymmetricalTransition):

    def __init__(self, sigma):
        super(Normal, self).__init__()
        self.sigma = sigma

    def log_prob(self, mean, conditionals):
        normal = NormalDistribution(mean, self.sigma)
        log_probabilities = normal.log_prob(conditionals)
        del Normal

        return log_probabilities

    def sample(self, means, samples=1):
        with torch.no_grad():
            means = means.view(-1, 1)
            samples = (torch.randn(means.size(0), samples) * self.sigma) + thetas)

        return samples


class MultivariateNormal(SymmetricalTransition):

    def __init__(self, sigma):
        super(MultivariateNormal, self).__init__()
        self.sigma = sigma
        self.dimensionality = sigma.size(0)

    def log_prob(self, mean, conditionals):
        normal = MultivariateNormalDistribution(mean, self.sigma)

        return normal.log_prob(conditionals)

    def sample(self, means, samples=1):
        x = []

        with torch.no_grad():
            means = means.view(-1, self.dimensionality)
            mean_samples = torch.Size([samples])
            for mean in means:
                normal = MultivariateNormalDistribution(mean, self.sigma)
                x.append(normal.sample(mean_samples).view(-1, samples, self.dimensionality))
            x = torch.cat(x, dim=0).squeeze()

        return x
