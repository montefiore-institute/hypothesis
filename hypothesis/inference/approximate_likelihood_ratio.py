"""
Approximate Likelihood Ratios.
"""

import torch
import hypothesis

from hypothesis.inference import Method
from hypothesis.util.constant import epsilon



def log_likelihood_ratio(classifier, observations, theta, theta_next):
    """
    Given a parameterized classifier, this method will compute the approximate log likelihood-ratio.

    Arguments:
        classifier: A ``ParameterizedClassifier``.
        observations: A tensor containing all observations.
        theta: Alternative hypothesis.
        theta_next: 0-hypothesis to test against.
        lower: Lower-bound of the search-space. Typically, this is the lower-bound in
               which the parameterized classifier has been trained.
        upper: Upper-bound of the search-space.

    .. note: This method assumes a ``ParameterizedClassifier``.
    """
    # Check if a lower and upper bound of the search space has been specified.
    lower = classifier.lower
    upper = classifier.upper
    if lower is not None and upper is not None:
        # Check if we are within the specified bounds.
        if (theta_next < lower).any() or (theta_next > upper).any():
            return torch.tensor(float("-inf"))
    num_observations = observations.size(0)
    # Prepare the classifier inputs.
    theta_next = theta_next.repeat(num_observations).view(num_observations, -1)
    theta = theta.repeat(num_observations).view(num_observations, -1)
    # Compute the approximate likelihood-raito.
    ratio_next = marginal_ratio(classifier, observations, theta_next).log().sum()
    ratio = marginal_ratio(classifier, observations, theta).log().sum()
    likelihood_ratio = (ratio_next - ratio).detach()

    return likelihood_ratio


def marginal_ratio(classifier, observations, thetas):
    r"""Computes the ratio p(x | theta) / p(x) given a set of observations."""
    s = classifier(observations, thetas)
    ratio = ((1 - s) / (s + epsilon))

    return ratio


class ClassifierMaximumLikelihoodOptimization(Method):

    KEY_THETA = "theta"
    KEY_STEPS = "steps"

    def __init__(self, classifier, batch_size=32, lr=0.001):
        super(ClassifierMaximumLikelihoodOptimization, self).__init__()
        self.batch_size = batch_size
        self.classifier = classifier
        self.lr = lr

    def infer(self, observations, **kwargs):
        theta = torch.tensor(kwargs[self.KEY_THETA]).float().view(-1).detach()
        steps = int(kwargs[self.KEY_STEPS])
        optimizer = torch.optim.Adam([theta], lr=self.lr)
        # TODO, Implement subsampling.
        for step in range(steps):
            optimizer.zero_grad()
            grad = self.classifier.grad_log_likelihood(observations, theta).expand_as(theta).detach()
            theta.grad = grad
            optimizer.step()

        return theta.detach()
