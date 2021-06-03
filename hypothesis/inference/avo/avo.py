r"""Implementation of Adversarial Variational Optimization

"""

import hypothesis as h
import numpy as np

from hypothesis.engine import Procedure


class AdversarialVariationalOptimization(Procedure):

    def __init__(self,
        discriminator,
        proposal,
        simulator,
        baseline=None,
        lr_discriminator=0.0001,
        lr_proposal=0.001,
        batch_size=32,
        r1_regularization=10.0):
        super(AdversarialVariationalOptimization, self).__init__()

    def _register_events(self):
        pass  # TODO Implement
