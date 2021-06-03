r"""Proposal distributions for Adversarial Variational Optimization.

"""

import torch

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal



class BaseAdversarialVariationalOptimizationProposal(torch.nn.Module):

    def __init__(self):
        super(BaseAdversarialVariationalOptimizationProposal, self).__init__()

    def clone(self):
        raise NotImplementedError

    def fix(self):
        raise NotImplementedError

    def log_prob(self, inputs):
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError

    def sample(self, size=(1,)):
        raise NotImplementedError


class NormalProposal(BaseAdversarialVariationalOptimizationProposal):

    def __init__(self, loc, scale):
        super(NormalProposal, self).__init__()
        self._loc = torch.tensor(loc).float()
        self._scale = torch.tensor(scale).float()
        self._loc.requires_grad = True
        self._scale.requires_grad = True
        self._parameters = [self._loc, self._scale]
        self._distribution = Normal(self._loc, self._scale)

    @torch.no_grad()
    def clone():
        loc = self._loc.clone()
        scale = self._loc.clone()

        return NormalProposal(loc, scale)

    @torch.no_grad()
    def fix(self):
        self._scale.abs_()

    def log_prob(self, inputs):
        inputs = inputs.view(-1)

        return self._distribution.log_prob(inputs)

    def parameters(self):
        return self._parameters

    def sample(self, size=(1,)):
        return self._distribution.sample(size)


class MultivariateNormalProposal(BaseAdversarialVariationalOptimizationProposal):

    def __init__(self, mean, sigma):
        super(MultivariateNormalProposal, self).__init__()
        self._mean = torch.tensor(mean).float()
        self._sigma = torch.tensor(sigma).float()
        self._mean.requires_grad = True
        self._sigma.requires_grad = True
        self._parameters = [self._mean, self._sigma]
        self._distribution = MultivariateNormal(self._mean, self._sigma)

    @torch.no_grad()
    def clone():
        mean = self._mean.clone()
        sigma = self._sigma.clone()

        return MultivariateNormalProposal(mean, sigma)

    @torch.no_grad()
    def fix(self):
        self._sigma.abs_()

    def log_prob(self, inputs):
        inputs = inputs.view(-1, len(self._mean))

        return self._distribution.log_prob(inputs)

    def parameters(self):
        return self._parameters

    def sample(self, size=(1,)):
        return self._distribution.sample(size)
