r"""Hypothesis colormap definitions.

"""

import matplotlib.pyplot as plt

from colour import Color
from matplotlib.colors import LinearSegmentedColormap


def make_linear_map(colors, name="colormap"):
    r"""Takes a list of hexademical color representations and
    builds a linearized colormap.
    """
    return LinearSegmentedColormap.from_list(name,[Color(c).rgb for c in colors])


### BEGIN Colormap definitions #################################################

cold_colors = ["#279AF1", "#FFFFFF"]
cold = make_linear_map(cold_colors, name="cold")
cold_colors_r = cold_colors[::-1]
cold_r = make_linear_map(cold_colors_r, name="cold_r")

### END Colormap definitions ###################################################
