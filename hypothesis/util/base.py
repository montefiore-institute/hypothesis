r"""General utilities for :mod:`hypothesis`."""

import torch


def is_iterable(item):
    r"""Checks whether the specified item is iterable.

    :param item: Any possible Python instance.
    :rtype: bool
    """
    return hasattr(item, "__getitem__")


def is_tensor(item):
    r"""Checks whether the specified item is a PyTorch tensor.

    :param item: Any possible Python instance.
    :rtype: bool
    """
    return torch.is_tensor(item)


def is_integer(item):
    r"""Checks whether the specified item is an integer.

    :param item: Any possible Python instance.
    :rtype: bool
    """
    try:
        integer = int(item)
        is_integer = True
    except:
        is_integer = False

    return is_integer


def load_module(full_modulename):
    r"""Loads the specified module (or class).

    :param full_modulename: The full module name of the method, class
                            or variable to load.
    """
    if full_modulename is None:
        raise ValueError("The specified modulename cannot be `None`.")
    module_name, name = full_modulename.rsplit('.', 1)
    module = __import__(module_name, fromlist=[name])

    return getattr(module, name)
