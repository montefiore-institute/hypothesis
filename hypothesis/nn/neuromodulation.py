import numpy as np
import torch
import torch.nn.functional as F



class BaseNeuromodulatedModule(torch.nn.Module):

    def __init__(self):
        super(BaseNeuromodulatedModule, self).__init__()

    def forward(self, x):
        raise NotImplementedError

    def update(self, context):
        raise NotImplementedError



class NeuromodulatedReLU(BaseNeuromodulatedModule):

    def __init__(self, controller, inplace=False):
        super(NeuromodulatedReLU, self).__init__()
        self.controller = controller
        self.slopes = None

    def forward(self, x):
        return F.relu(self.slopes + x)

    def update(self, context):
        self.slopes = self.controller(context)
