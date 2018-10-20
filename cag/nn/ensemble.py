"""
Ensemble module.
"""

import copy
import torch

from cag.util import initialize_parameters



class Ensemble(torch.nn.Module):

    def __init__(self, module, num_modules=1):
        super(Ensemble, self).__init__()
        self._num_modules = num_modules
        self._modules = torch.nn.ModuleList([module] * num_modules)
        self._initialize_modules()

    def _initialize_modules(self):
        for module in self._modules:
            module.apply(initialize_parameters)

    def forward(self, x):
        y_hats = []

        for module in self._modules:
            y_hat = module(x)
            y_hats.append(y_hat)
        y_hat = torch.cat(y_hats, dim=0).mean(dim=0)

        return y_hat
