import numpy as np
import torch

from hypothesis.metric import BaseMetric



class ExponentialAverage(BaseMetric):
    r""""""

    def __init__(self, initial_value=None, decay=.95):
        self.decay = decay
        self.initial_value = initial_value
        self.current_value = initial_value
        if initial_value is not None:
            self.history = [self.current_value]

    def update(self, value):
        next_value = self.decay * value + (1 - self.decay) * self.current_value
        self.history.append(next_value)
        self.current_value = next_value

    def reset(self):
        self.current_value = self.initial_value
        if initial_value is not None:
            self.history = [self.initial_value]

    def __getitem__(self, pattern):
        return self.history[pattern]

    def __len__(self):
        return len(self.history)
