"""
Baselines for variance reduction in AVO.
"""

import torch



class Baseline:

    def apply(self, gradients, x):
        raise NotImplementedError


class NashBaseline(Baseline):

    def __init__(self, discriminator):
        self._discriminator = discriminator
        self._constant = torch.tensor(.5).log().detach()

    def apply(self, gradients, x):
        baselines = []

        with torch.no_grad():
            y = (1 - self._discriminator(x)).log()
            b = (self._constant - y)
            for g in gradients[0]:
                baselines.append(b)
            baselines = torch.cat(baselines, dim=1)

        return baselines


class MeanBaseline(Baseline):

    def __init__(self, discriminator):
        self._discriminator = discriminator

    def apply(self, gradients, x):
        baselines = []

        with torch.no_grad():
            y = (1 - self._discriminator(x)).log()
            b = (y.mean() - y)
            for g in gradients[0]:
                baselines.append(b)
            baselines = torch.cat(baselines, dim=1)

        return baselines
