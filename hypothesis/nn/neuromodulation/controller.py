import hypothesis
import numpy as np
import torch

from hypothesis.nn import MLP



def allocate_default_neuromodulation_controller(shape_context,
    activation=hypothesis.default.activation,
    dropout=hypothesis.default.dropout,
    layers=hypothesis.default.trunk):
    return DefaultNeuromodulationController(
        shape_context=shape_context,
        activation=activation,
        dropout=dropout,
        layers=layers)



class DefaultNeuromodulationController(torch.nn.Module):

    def __init__(self, shape_context,
        activation=hypothesis.default.activation,
        dropout=hypothesis.default.dropout,
        layers=hypothesis.default.trunk):
        super(DefaultNeuromodulationController, self).__init__()
        self.mlp = MLP(
            shape_xs=shape_context,
            shape_ys=(1,),
            activation=activation,
            dropout=dropout,
            layers=layers,
            transform_output=None)

    def forward(self, x):
        return self.mlp(x)
