r""""""

import glob
import numpy as np
import torch



def load_losses(pattern, format="numpy"):
    r""""""
    formats = {
        "numpy": load_losses_numpy,
        "torch": load_losses_torch}
    # Check if the specified format is available.
    if format not in formats.keys():
        raise ValueError("The format", format, "is not supported.")
    paths = glob.glob(pattern)
    with torch.no_grad():
        losses = formats[format](paths)

    return losses


def load_losses_numpy(paths):
    r""""""
    losses = []

    for path in paths:
        losses.append(torch.from_numpy(np.load(path)).view(1, -1))

    return losses


def load_losses_torch(paths):
    r""""""
    losses = []

    for path in paths:
        losses.append(torch.load(path, map_location="cpu").view(1, -1))

    return losses


def stack_losses(losses):
    r""""""
    return torch.cat(losses, dim=0)


def load_and_stack_losses(pattern, format="numpy"):
    r""""""
    with torch.no_grad():
        losses = load_losses(pattern, format=format)
        stacked = stack_losses(losses)

    return stacked
