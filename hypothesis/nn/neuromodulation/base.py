import hypothesis
import torch



class BaseNeuromodulatedModule(torch.nn.Module):

    def __init__(self, controller, activation=hypothesis.default.activation, **kwargs):
        super(BaseNeuromodulatedModule, self).__init__()
        self.activation = activation(**kwargs)
        self.bias = None
        self.controller = controller

    def forward(self, x, context=None):
        if context is None:
            self.update(context)

        return self.activation(x + self.bias)

    def update(self, context):
        self.bias = self.controller(context)
