r"""Proposal distributions for Adversarial Variational Optimization.

"""

import torch

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal



class AdversarialVariationalOptimizationProposal(torch.nn.Module):

    def __init__(self):
        super(AdversarialVariationalOptimizationProposal, self).__init__()

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


class NormalProposal(AdversarialVariationalOptimizationProposal):

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
