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

    def procedure(self, observations, **kwargs):
        raise NotImplementedError

    def infer(self, observations, **kwargs):
        self.start()
        self.fire_event(event.start)
        result = self.procedure(observations, **kwargs)
        self.fire_event(event.terminate)
        self.terminate()

        return result


class SimulatorMethod(Method):

    def __init__(self, simulator):
        super(SimulatorMethod, self).__init__()
        self.simulator = simulator

    def procedure(self, observations, **kwargs):
        raise NotImplementedError
