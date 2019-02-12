"""
Parameterized classifier.
"""

import torch
import hypothesis

from hypothesis.util import epsilon
from hypothesis.inference.approximate_likelihood_ratio import log_likelihood_ratio
from hypothesis.inference.approximate_likelihood_ratio import marginal_ratio
from torch.multiprocessing import Process



class AbstractParameterizedClassifier(torch.nn.Module):

    def __init__(self, lower=None, upper=None):
        super(AbstractParameterizedClassifier, self).__init__()
        # Define the lower bound.
        if lower is not None:
            self.lower = torch.tensor(lower).view(-1).float().detach()
        else:
            self.lower = None
        # Define the upper bound.
        if upper is not None:
            self.upper = torch.tensor(upper).view(-1).float().detach()
        else:
            self.upper = None

    def marginal_ratio(self, observations, theta):
        return marginal_ratio(self, observations, theta)

    def log_likelihood_ratio(self, observations, theta, theta_next):
        return log_likelihood_ratio(self, observations, theta, theta_next)

    def grad_log_likelihood(self, observations, theta):
        num_observations = observations.size(0)
        thetas = theta.repeat(num_observations).view(num_observations, -1).detach()
        thetas.requires_grad = True
        ratios = self.marginal_ratio(observations, thetas)
        torch.autograd.backward(ratios.split(1), None)
        gradient = (-thetas.grad / (ratios + epsilon)).sum(dim=0).detach()

        return gradient

    def forward(self, observations, thetas):
        raise NotImplementedError



class ParameterizedClassifier(AbstractParameterizedClassifier):

    def __init__(self, classifier, lower=None, upper=None):
        super(ParameterizedClassifier, self).__init__(lower, upper)
        self.classifier = classifier

    def parameters(self):
        return self.classifier.parameters()

    def forward(self, observations, thetas):
        n = thetas.size(0)
        thetas = thetas.view(n, -1)
        observations = observations.view(n, -1)
        x = torch.cat([observations, thetas], dim=1)

        return self.classifier(x)



class ParameterizedClassifierEnsemble(torch.nn.Module):

    def __init__(self, classifiers):
        super(ParameterizedClassifierEnsemble, self).__init__()
        self.classifiers = classifiers

    def marginal_ratio(self, observations, theta):
        ratios = []
        for classifier in self.classifiers:
            ratios.append(classifier.marginal_ratio(observations, theta).view(-1))
        ratios = torch.cat(ratios, dim=0)

        return ratios.mean().squeeze()

    def log_likelihood_ratio(self, observations, theta, theta_next):
        ratios = []
        for classifier in self.classifiers:
            ratios.append(classifier.log_likelihood_ratio(observations, theta, theta_next).view(-1))
        ratios = torch.cat(ratios, dim=0)

        return ratios.mean().squeeze()

    def grad_log_likelihood(self, observations, theta):
        grads = []
        for classifier in self.classifiers:
            grads.append(classifier.grad_log_likelihood(observations, theta))
        grads = torch.cat(grads, dim=0)

        return grads.mean(dim=0)

    def forward(self, observations, thetas):
        outputs = []
        for classifier in self.classifiers:
            outputs.append(classifier(observations, thetas))
        output = torch.cat(outputs, dim=0).mean(dim=0)

        return output
