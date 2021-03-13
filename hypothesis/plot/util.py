r"""Plotting utilities for Hypothesis.

"""


def make_square(ax):
    r"""Makes the `matplotlib` axes square, irrespective of the data limits.

    """
    dr = ax.get_data_ratio()
    ax.set_aspect(1.0 / dr, adjustable="box")
