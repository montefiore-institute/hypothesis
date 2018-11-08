"""
Methods abstraction.
"""

import numpy as np
import torch

from hypothesis.engine import Module
from hypothesis.engine import event



class Method(Module):

    def __init__(self):
        super(Method, self).__init__()

    def procedure(self, x, **kwargs):
        raise NotImplementedError

    def sample(self, **kwargs):
        self.start()
        self.fire_event(event.start)
        result = self.procedure(**kwargs)
        self.fire_event(event.terminate)
        self.terminate()

        return result
