Module hypothesis.nn.model.mlp.base
===================================
Base model of a multi-layered perceptron (MLP).

Classes
-------

`MLP(shape_xs, shape_ys, activation=torch.nn.modules.activation.LeakyReLU, dropout=0.0, layers=(256, 256, 256), transform_output='normalize')`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    
    :ivar training: Boolean represents whether this module is in training or
                    evaluation mode.
    :vartype training: bool
    
    Initializes a multi-layered perceptron (MLP).
    
    :param shape_xs: A tuple describing the shape of the MLP inputs.
    :param shape_ys: A tuple describing the shape of the MLP outputs.
    :param activation: An allocator which, when called,
                       returns a :mod:`torch` activation.
    :param dropout: Dropout rate.
    :param transform_output: Output transformation.
    
    :rtype: :class:`hypothesis.nn.model.mlp.MLP`

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :

    ### Methods

    `forward(self, xs) ‑> Callable[..., Any]`
    :   Defines the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.