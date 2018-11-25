"""
Ensemble methods for hypothesis.
"""

import torch



class Ensemble(torch.nn.Module):

    def __init__(self, models):
        super(Ensemble, self).__init__()
        self._models = models

    def forward(self, x):
        y = self._models[0](x)
        for model in self._models[1:]:
            y += model(x)
        y /= len(self._models)

        return y
