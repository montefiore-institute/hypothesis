r"""General utilities for :mod:`hypothesis`."""


def is_iterable(item):
    r"""Checks whether the specified item is iterable.

    :param item: Any possible Python instance.
    :rtype: bool
    """
    return hasattr(item, "__getitem__")


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
