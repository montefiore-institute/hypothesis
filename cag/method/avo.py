"""
Adversarial Variational Optimization
"""

import cag.baseline
import cag.method



class AdversarialVariationalOptimization(Method):

    def __init__(self, simulator, discriminator, proposal, baseline=None):
        super(self, AdversarialVariationalOptimization).__init__(simulator)
        self.discriminator = discriminator
        self.proposal = proposal

    def _reset(self):
        raise NotImplementedError

    def infer(self, x_o, num_iterations=1000):
        self._reset()

        raise NotImplementedError
