"""
Parameterized classifier.
"""

import torch
import hypothesis

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

    def parameters(self):
        raise NotImplementedError

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
        x = torch.cat([thetas, observations], dim=1)

        return self.classifier(x)
