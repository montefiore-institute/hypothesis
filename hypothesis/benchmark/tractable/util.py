r"""Utilities for the tractable benchmark.

"""

import torch

from torch.distributions.multivariate_normal import MultivariateNormal as Normal



def allocate_prior():
    lower = -3 * torch.ones(5).float()
    upper = 3 * torch.ones(5).float()

    return Uniform(lower, lower)


def allocate_truth():
    truth = [0.7, -2.9, -1.0, -0.9, 0.6]

    return torch.tensor(truth).float()


def log_likelihood(theta, x):
    with torch.no_grad():
        input = theta
        mean = torch.tensor([input[0], input[1]])
        scale = 1.0
        s_1 = input[2] ** 2
        s_2 = input[3] ** 2
        rho = input[4].tanh()
        covariance = torch.tensor([
            [scale * s_1 ** 2, scale * rho * s_1 * s_2],
            [scale * rho * s_1 * s_2, scale * s_2 ** 2]])
        normal = Normal(mean, covariance)
        m = x.view(-1, 2)
        log_likelihood = normal.log_prob(m).sum()

    return log_likelihood



class Uniform(torch.distributions.uniform.Uniform):

    def __init__(self, lower, upper):
        super(Uniform, self).__init__(lower, upper)

    def log_prob(self, sample):
        return super(Uniform, self).log_prob(sample).mean()
