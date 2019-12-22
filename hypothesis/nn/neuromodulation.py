import hypothesis
import numpy as np
import torch



class BaseNeuromodulatedModule(torch.nn.Module):

    def __init__(self):
        super(BaseNeuromodulatedModule, self).__init__()

    def forward(self, x, context=None):
        raise NotImplementedError

    def update(self, context):
        raise NotImplementedError



class NeuromodulatedELU(BaseNeuromodulatedModule):

    def __init__(self, controller, inplace=False):
        super(NeuromodulatedELU, self).__init__()
        self.activation = torch.nn.ELU(inplace=inplace)
        self.controller = controller
        self.bias = None

    def forward(self, x, context=None):
        if context is not None:
            self.update(context)
        return self.activation(x + self.bias)

    def update(self, context):
        self.bias = self.controller(context)



class NeuromodulatedReLU(BaseNeuromodulatedModule):

    def __init__(self, controller, inplace=False):
        super(NeuromodulatedReLU, self).__init__()
        self.activation = torch.nn.ReLU(inplace=inplace)
        self.controller = controller
        self.bias = None

    def forward(self, x, context=None):
        if context is not None:
            self.update(context)
        return self.activation(x + self.bias)

    def update(self, context):
        self.bias = self.controller(context)



class NeuromodulatedTanh(BaseNeuromodulatedModule):

    def __init__(self, controller, inplace=False):
        super(NeuromodulatedTanh, self).__init__()
        self.activation = torch.nn.Tanh(inplace=inplace)
        self.controller = controller
        self.bias = None

    def forward(self, x, context=None):
        if context is not None:
            self.update(context)
        return self.activation(x + self.bias)

    def update(self, context):
        self.bias = self.controller(context)
