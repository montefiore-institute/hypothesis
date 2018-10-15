"""
Proposals.

TODO Write doc.
"""

import torch

from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.uniform import Uniform

from cag.util import sample_distribution



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


class TruncatedProposal(Proposal):

    def __init__(self, proposal, min_bound, max_bound):
        self._min_bound = torch.tensor(min_bound).float()
        self._max_bound = torch.tensor(max_bound).float()
        self._proposal = proposal

    def clone(self):
        with torch.no_grad():
            proposal = TruncatedProposal(self._propoxal, self._min_bound, self._max_bound)

        return proposal

    def fix(self):
        self._propoxal.fix()

    def log_prob(self, thetas):
        return self._proposal.log_prob(thetas)

    def parameters(self):
        return self._proposal.parameters()

    def sample(self, num_samples):
        # TODO Ensure samples between bounds.
        raise NotImplementedError


class UniformProposal(Proposal):

    def __init__(self, min_bound, max_bound):
        self._min_bound = torch.tensor(min_bound).float()
        self._min_bound.requires_grad = True
        self._max_bound = torch.tensor(max_bound).float()
        self._max_bound.requires_grad = True
        self._distribution = Uniform(low=self._min_bound, high=self._max_bound)
        self._parameters = [self._min_bound, self._max_bound]

    def clone(self):
        with torch.no_grad():
            proposal = UniformProposal(self._min_bound, self._max_bound)

        return proposal

    def fix(self):
        pass

    def log_prob(self, thetas):
        return self._distribution.log_prob(thetas)

    def parameters(self):
        return self._parameters

    def sample(self, num_samples):
        return sample_distribution(self._distribution, num_samples)


class NormalProposal(Proposal):

    def __init__(self, mu=0., sigma=1.):
        self._mu = torch.tensor(mu).float()
        self._mu.requires_grad = True
        self._sigma = torch.tensor(sigma).float()
        self._sigma.requires_grad = True
        self._parameters = [self._mu, self._sigma]
        self._distribution = Normal(self._mu, self._sigma)

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

    def __init__(self, mu, sigma, mask=torch.eye(mu.size(0))):
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
