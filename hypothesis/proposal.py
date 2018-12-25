"""
Proposals
"""

import torch

from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.uniform import Uniform

from hypothesis.util import sample_distribution



class Proposal:

    def clone(self):
        raise NotImplementedError

    def fix(self):
        raise NotImplementedError

    def log_prob(self, thetas):
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError

    def sample(self, num_samples):
        raise NotImplementedError


class NormalProposal(Proposal):

    def __init__(self, mu=0., sigma=1.):
        self._distribution = Normal(mu, sigma)
        self._mu = self._distribution.loc
        self._mu.requires_grad = True
        self._sigma = self._distribution.scale
        self._sigma.requires_grad = True
        self._parameters = [self._mu, self._sigma]

    def clone(self):
        with torch.no_grad():
            proposal = NormalProposal(self._mu, self._sigma)

        return proposal

    def fix(self):
        with torch.no_grad():
            self._sigma.abs_()

    def log_prob(self, thetas):
        return self._distribution.log_prob(thetas)

    def parameters(self):
        return self._parameters

    def sample(self, num_samples):
        return sample_distribution(self._distribution, num_samples)


class MultivariateNormalProposal(Proposal):

    def __init__(self, mu, sigma):
        self._mu = torch.tensor(mu).float()
        self._mu.requires_grad = True
        self._sigma = torch.tensor(sigma).float()
        self._sigma.requires_grad = True
        self._parameters = [self._mu, self._sigma]
        self._distribution = MultivariateNormal(self._mu, self._sigma)

    def clone(self):
        with torch.no_grad():
            proposal = MultivariateNormalProposal(self._mu, self._sigma)

        return proposal

    def fix(self):
        with torch.no_grad():
            self._sigma.abs_()

    def log_prob(self, thetas):
        return self._distribution.log_prob(thetas)

    def parameters(self):
        return self._parameters

    def sample(self, num_samples):
        thetas = sample_distribution(self._distribution, num_samples)

        return thetas


class MaskedMultivariateNormalProposal(Proposal):

    def __init__(self, mu, sigma, mask):
        self._mu = torch.tensor(mu).float()
        self._mu.requires_grad = True
        self._sigma = torch.tensor(sigma)
        self._sigma.requires_grad = True
        self._mask = torch.tensor(mask).float()
        self._parameters = [self._mu, self._sigma]
        self._distribution = MultivariateNormal(self._mu, self._sigma)

    def clone(self):
        with torch.no_grad():
            proposal = MaskedMultivariateNormalProposal(self._mu, self._sigma, self._mask)

        return proposal

    def fix(self):
        with torch.no_grad():
            self._sigma.abs_()
            self._sigma.mul_(self.mask)

    def log_prob(self, thetas):
        return self._distribution.log_prob(thetas)

    def parameters(self):
        return self._parameters

    def sample(self, num_samples):
        thetas = sample_distribution(self._distribution, num_samples)

        return thetas
