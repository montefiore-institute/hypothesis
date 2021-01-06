r"""Utilies for :mod:`hypothesis.nn`.

"""

import torch

from hypothesis.util import is_iterable


def allocate_output_transform(transformation, shape):
    r"""Allocates the specified output transformation for the given output shape.

    :param transformation: ``"normalize"`` or an allocator defining the
                           transformation.
    :param shape: Output shape of the transformation.
    :type shape: iterable of ints
    """
    if is_iterable(shape):
        dim = dimensionality(shape)
    else:
        dim = shape
    if transformation == "normalize":
        if dim > 1:
            mapping = torch.nn.Softmax(dim=0)
        else:
            mapping = torch.nn.Sigmoid()
    elif transformation is not None:
        mapping = transformation()
    else:
        mapping = None

    return mapping


def dimensionality(shape):
    r"""Computes the total dimensionality of the specified shape.

    :param shape: Tuple describing the shape of the expected data.
    :type shape: iterable of ints
    :rtype: int
    """
    assert(is_iterable(shape))

    dimensionality = 1
    for dim in shape:
        dimensionality *= dim

    return dimensionality
