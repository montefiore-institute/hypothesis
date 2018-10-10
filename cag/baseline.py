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
        with torch.no_grad():
            batch_size = x.size(0)
            numerators = []
            denominators = []
            num_parameters = len(gradients[0])
            for p in gradients[0]:
                numerators.append(torch.zeros_like(p))
                denominators.append(torch.zeros_like(p))
            y = (1 - self._discriminator(x)).log()
            for index, gradient in enumerate(gradients):
                for pg_index, pg in enumerate(gradients[index]):
                    pg2 = pg.pow(2)
                    y_theta = y[index].squeeze()
                    numerators[pg_index] += pg2 * y_theta
                    denominators[pg_index] += pg2
            b = []
            for index in range(num_parameters):
                numerators[index] /= batch_size
                denominators[index] /= batch_size
                b.append(numerators[index] / denominators[index])
            baselines = []
            for index in range(batch_size):
                parameters = []
                for p_index in range(num_parameters):
                    p = (b[p_index] - y[index])
                    parameters.append(p)
                baselines.append(torch.tensor(parameters).view(1, -1))
            baselines = torch.cat(baselines, dim=0)

        return baselines
