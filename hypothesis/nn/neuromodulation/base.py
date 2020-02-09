import hypothesis
import torch

from hypothesis.nn.util import list_modules_with_type



def allocate_neuromodulated_activation(activation, allocator):
    class LambdaNeuromodulatedActivation(BaseNeuromodulatedModule):

        def __init__(self):
            super(LambdaNeuromodulatedActivation, self).__init__(
                controller=allocator(),
                activation=activation)

    return LambdaNeuromodulatedActivation


def list_neuromodulated_modules(module):
    desired_type = BaseNeuromodulatedModule

    return list_modules_with_type(module, desired_type)



class BaseNeuromodulatedModule(torch.nn.Module):

    def __init__(self, controller, activation=hypothesis.default.activation, **kwargs):
        super(BaseNeuromodulatedModule, self).__init__()
        self.activation = activation(**kwargs)
        self.bias = None
        self.controller = controller

    def forward(self, x, context=None):
        if context is not None:
            self.update(context)

        return self.activation(x + self.bias)

    def update(self, context):
        self.bias = self.controller(context)
