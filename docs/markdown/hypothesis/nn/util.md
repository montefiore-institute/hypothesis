Module hypothesis.nn.util
=========================
Utilies for :mod:`hypothesis.nn`.

Functions
---------

    
`allocate_output_transform(transformation, shape)`
:   Allocates the specified output transformation for the given output shape.
    
    :param transformation: ``"normalize"`` or an allocator defining the
                           transformation.
    :param shape: Output shape of the transformation.
    :type shape: iterable of ints

    
`dimensionality(shape)`
:   Computes the total dimensionality of the specified shape.
    
    :param shape: Tuple describing the shape of the expected data.
    :type shape: iterable of ints
    :rtype: int