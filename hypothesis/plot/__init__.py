r"""Base plotting utilities and definition of ``matplotlib`` theme.


"""

import matplotlib.pyplot as plt

from .util import *



def activate():
    r"""Enables to Hypothesis plotting style by default."""
    plt.style.use("hypothesis")


def deactivate():
    r"""Disables the Hypothesis plotting style by reverting
    to Matplotlib's default.
    """
    plt.style.use("default")


class HypothesisPlottingStyle:
    r"""Decorator to enable the Hypothesis matplotlib style
    in a specific context.

    To be used as

        import hypothesis as h
        import matplotlib.pyplot as plt

        with h.plot.style:
            plt.plot([1, 2], [1, 2])
            plt.show()
    """

    def __enter__(self):
        plt.style.use("hypothesis")

        return plt

    def __leave__(self):
        plt.style.use("default")

    def __exit__(self, exc_type, exc_value, traceback):
        self.__leave__()


style = HypothesisPlottingStyle()
