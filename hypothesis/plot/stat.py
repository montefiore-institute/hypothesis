r"""Common plotting utilities for inference.

"""

import torch

from .util import *


@torch.no_grad()
def pdf(ax, extent, pdf, **kwargs):
    pdf = np.squeeze(pdf)
    d = len(pdf.shape)
    supported_dimensionalities = {
        1: pdf_1d,
        2: pdf_2d}
    if d not in supported_dimensionalities.keys():
        raise NotImplementedError("Plotting PDF's with a dimensionality of", d, "are not supported.")
    supported_dimensionalities[d](ax, extent, pdf, **kwargs)
    make_square(ax)


@torch.no_grad()
def pdf_1d(ax, extent, pdf, **kwargs):
    ax.plot(extent, pdf)
    ax.set_ylabel(r"Posterior density $p(\vartheta\vert x)$")


@torch.no_grad()
def pdf_2d(ax, extent, pdf, **kwargs):
    raise NotImplementedError
