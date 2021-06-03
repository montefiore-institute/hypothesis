r"""Baselines for variance reduction in Adversarial Variational Optimization.

"""

import torch



class BaseBaseline:

    def apply(self, gradients, observables):
        raise NotImplementedError


class MeanBaseline(BaseBaseline):

    def __init__(self, discriminator):
        self._discriminator = discriminator

    @torch.no_grad()
    def apply(self, gradients, observables):
        baselines = []

        d = (1 - self._discriminator(observables)).log()
        b = (d.mean() - d)

        return b
