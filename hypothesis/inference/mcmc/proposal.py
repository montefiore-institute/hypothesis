r"""Definition of MCMC proposal (transition) distributions.

"""

import hypothesis as h
import torch

from hypothesis.util import is_tensor
from torch.distributions.multivariate_normal import MultivariateNormal as MultivariateNormalDistribution
from torch.distributions.normal import Normal as NormalDistribution



class BaseProposal:
    r""""""

    def log_prob(self, inputs, conditionals):
        raise NotImplementedError

    def sample(self, inputs, samples=1):
        raise NotImplementedError

    def is_symmetrical(self):
        raise NotImplementedError


class SymmetricalProposal(BaseProposal):

    def is_symmetrical(self):
        return True


class AsymmetricalProposal(BaseProposal):

    def is_symmetrical(self):
        return False


class Normal(SymmetricalProposal):

    def __init__(self, sigma):
        super(NormalProposal, self).__init__()
        self._sigma = sigma

    def log_prob(self, inputs, conditionals):
        inputs = inputs.view(-1, 1)
        normals = NormalDistribution(inputs, self._sigma)
        log_probabilities = normals.log_prob(conditionals)

        return log_probabilities

    @torch.no_grad()
    def sample(self, inputs, samples=1):
        inputs = inputs.view(-1, 1)
        samples = torch.randn(inputs.size(0), samples)
        samples = samples.to(inputs.device)

        return (samples * self._sigma) + inputs


class MultivariateNormal(SymmetricalProposal):

    def __init__(self, sigma):
        super(MultivariateNormal, self).__init__()
        self._sigma = sigma
        self._dimensionality = sigma.size(0)

    def log_prob(self, inputs, conditionals):
        normal = MultivariateNormalDistribution(inputs, self._sigma)

        return normal.log_prob(conditionals)

    @torch.no_grad()
    def sample(self, inputs, samples=1):
        xs = []

        inputs = inputs.view(-1, self._dimensionality)
        for mu in means:
            n = MultivariateNormalDistribution(mu, self._sigma)
            xs.append(n.sample((samples,)).view(-1, samples, self._dimensionality))

        return torch.cat(xs, dim=0).squeeze()
