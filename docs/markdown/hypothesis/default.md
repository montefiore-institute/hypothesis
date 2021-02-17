Module hypothesis.default
=========================
Default settings in Hypothesis.

Variables
---------

    
`batch_size`
:   Default batch size.

    
`dataloader_workers`
:   Default number of dataloader workers.

    
`dependent_delimiter`
:   Split character indicating the dependence between random variables.

    
`dropout`
:   Default dropout setting.

    
`epochs`
:   Default number of data epochs.

    
`independent_delimiter`
:   Split character indicating the independene between random variables.

    
`output_transform`
:   Default output transformation for neural networks.
    
    For 1-dimensional outputs, this is equivalent to torch.nn.Sigmoid.
    Otherwise, this will reduce to torch.nn.Softmax.

    
`trunk`
:   Default trunk of an MLP.

Classes
-------

`activation(negative_slope: float = 0.01, inplace: bool = False)`
:   Applies the element-wise function:
    
    .. math::
        \text{LeakyReLU}(x) = \max(0, x) + \text{negative\_slope} * \min(0, x)
    
    
    or
    
    .. math::
        \text{LeakyRELU}(x) =
        \begin{cases}
        x, & \text{ if } x \geq 0 \\
        \text{negative\_slope} \times x, & \text{ otherwise }
        \end{cases}
    
    Args:
        negative_slope: Controls the angle of the negative slope. Default: 1e-2
        inplace: can optionally do the operation in-place. Default: ``False``
    
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    
    .. image:: ../scripts/activation_images/LeakyReLU.png
    
    Examples::
    
        >>> m = nn.LeakyReLU(0.1)
        >>> input = torch.randn(2)
        >>> output = m(input)
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `inplace: bool`
    :

    `negative_slope: float`
    :

    ### Methods

    `extra_repr(self) ‑> str`
    :   Set the extra representation of the module
        
        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.

    `forward(self, input: torch.Tensor) ‑> torch.Tensor`
    :   Defines the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.