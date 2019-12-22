r""""""

import hypothesis
import matplotlib.pyplot as plt



def make_square(ax):
    set_aspect(ax, 1)


def set_aspect(ax, aspect):
    r""""""
    aspect = float(aspect)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    aspect = (x1 - x0) / (aspect * (y1 - y0))
    ax.set_aspect(aspect)
