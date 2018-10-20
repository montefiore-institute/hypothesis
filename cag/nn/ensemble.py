"""
Ensemble module.
"""

import copy
import torch

from cag.util import initialize_parameters



class Ensemble(torch.nn.Module):

    def __init__(self, model_allocator, num_models=1):
        super(Ensemble, self).__init__()
        self._num_models = num_models
        models = []
        for model_index in range(num_models):
            model = model_allocator()
            models.append(model)
        self._models = torch.nn.ModuleList(models)

    def forward(self, x):
        y_hats = []

        for module in self._models:
            y_hat = module(x)
            y_hats.append(y_hat)
        y_hat = torch.cat(y_hats, dim=1).mean(dim=1).view(-1, 1)

        return y_hat
