r"""Hypothesis workflow decorators.

"""

import hypothesis as h


def test(f):
    print("Testing!")
    return f


def precondition(f, condition=None):
    return f


def postcondition(f, condition=None):
    return f
