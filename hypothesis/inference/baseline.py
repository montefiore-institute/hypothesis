"""
Baselines for variance reduction in REINFORCE estimates.
"""

import torch



class Baseline:

    def apply(self, **kwargs):
        raise NotImplementedError


class NashBaseline(Baseline):

    def __init__(self, discriminator):
        self._discriminator = discriminator
        self._equilibrium = torch.tensor(.5).log().detach()

    def apply(self, **kwargs):
        baselines = []
        gradients = kwargs["gradients"]
        x = kwargs["x"]

        with torch.no_grad():
            y = (1 - self._discriminator(x)).log()
            b = (self._equilibrium - y)
            for g in gradients[0]:
                baselines.append(b)
            baselines = torch.cat(baselines, dim=1)

        return baselines


class MeanBaseline(Baseline):

    def __init__(self, discriminator):
        self._discriminator = discriminator

    def apply(self, **kwargs):
        baselines = []
        gradients = kwargs["gradients"]
        x = kwargs["x"]

        with torch.no_grad():
            y = (1 - self._discriminator(x)).log()
            b = (y.mean() - y)
            for g in gradients[0]:
                baselines.append(b)
            baselines = torch.cat(baselines, dim=1)

        return baselines


class AVOBaseline(Baseline):

    def __init__(self, discriminator):
        self._discriminator = discriminator

    def apply(self, **kwargs):
        numerators = []
        denominators = []
        gradients = kwargs["gradients"]
        x = kwargs["x"]
        num_parameters = len(gradients[0])
        batch_size = x.size(0)

        with torch.no_grad():
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
                b.append(numerators[index] / (denominators[index] + 10e-10))
            baselines = []
            for index in range(batch_size):
                parameters = []
                for p_index in range(num_parameters):
                    p = (b[p_index] - y[index])
                    parameters.append(p)
                baselines.append(parameters)

        return baselines
