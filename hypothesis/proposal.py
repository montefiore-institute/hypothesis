import torch

from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.uniform import Uniform



class Proposal:
    r"""Abstract base class for a proposal with parameters $\theta$."""

    def clone(self):
        raise NotImplementedError

    def log_prob(self, thetas):
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError

    def sample(self, num_samples):
        raise NotImplementedError

    def fix(self):
        pass



class NormalProposal(Proposal):

    def __init__(self, mu=0, sigma=1):
        self.mu = torch.tensor(mu).float().squeeze().detach()
        self.sigma = torch.tensor(sigma).float().squeeze().detach()
        self.distribution = Normal(self.mu, self.sigma)
        self.mu.requires_grad = True
        self.sigma.requires_grad = True
        self._parameters = [self.mu, self.sigma]

    def clone(self):
        return NormalProposal(self.mu.item(), self.sigma.item())

    def fix(self):
        with torch.no_grad():
            self.sigma.abs_()

    def log_prob(self, thetas):
        return self.distribution.log_prob(thetas).view(-1)

    def parameters(self):
        return self._parameters

    def sample(self, num_samples):
        return self.distribution.sample(torch.Size([num_samples]))



class MultivariateNormalProposal(Proposal):

    def __init__(self, mu, sigma):
        self.mu = torch.tensor(mu).float()
        self.mu.requires_grad = True
        self.sigma = torch.tensor(sigma).float()
        self.sigma.requires_grad = True
        self.distribution = MultivariateNormal(self.mu, self.sigma)
        self._parameters = [self.mu, self.sigma]

    def clone(self):
        return MultivariateNormalProposal(self.mu.detach(), self.sigma.detach())

    def fix(self):
        with torch.no_grad():
            self.sigma.abs_()

    def log_prob(self, thetas):
        return self.distribution.log_prob(thetas)

    def parameters(self):
        return self._parameters

    def sample(self, num_samples):
        return self.distribution.sample(torch.Size([num_samples]))
