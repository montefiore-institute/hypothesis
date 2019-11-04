import numpy as np
import torch

from hypothesis.metric import BaseValueMetric



class ExponentialAverageMetric(BaseValueMetric):
    r""""""

    def __init__(self, initial_value=None, decay=.99):
        super(ExponentialAverageMetric, self).__init__(initial_value)
        self.decay = decay

    def update(self, value):
        # Check if the current value was initialized.
        if self.current_value is not None:
            next_value = self.decay * value + (1 - self.decay) * self.current_value
        else:
            next_value = value
        self._set_current_value(value)
