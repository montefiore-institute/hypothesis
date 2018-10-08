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


class OptimalBaseline(Baseline):

    def __init__(self, discriminator):
        self._discriminator = discriminator

    def apply(self, gradients, x):
        baselines = []

        with torch.no_grad():
            batch_size = gradients.size(0)
            numerator = torch.zeros_like(gradients[0])
            denominator = torch.zeros_like(gradients[0])
            y = (1 - self._discriminator(x)).log()
            for index, gradient in enumerate(gradients):
                for pg_index, pg in enumerate(gradients[index]):
                    pg2 = pg.pow(2)
                    y_theta = y[index]
                    numerator[pg_index] += pg2 * y_theta
                    denominator[pg_index] += pg2
            numerator /= batch_size
            denominator /= batch_size
            b = numerator / denominator
            # TODO Implement

        return baselines
