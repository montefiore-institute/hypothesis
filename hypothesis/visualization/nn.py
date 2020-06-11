r""""""

import matplotlib.pyplot as plt
import numpy as np
import torch

from hypothesis.visualization.util import make_square
from hypothesis.visualization.util import set_aspect



@torch.no_grad()
def stack(paths):
    data = []
    for path in paths:
        data.append(np.load(path).reshape(1, -1))

    return torch.from_numpy(np.vstack(data))



@torch.no_grad()
def plot(paths, title=None):
    # Prepare the data
    data = stack(paths)
    figure, ax = plt.subplots(1)
    mean = data.mean(dim=0)
    std = data.std(dim=0)
    # Plot the data
    epochs = np.arange(1, len(mean) + 1)
    ax.set_title(title)
    ax.plot(epochs, mean, lw=2, color="black")
    ax.fill_between(epochs, mean - std, mean + std, color="black", alpha=0.1)
    ax.minorticks_on()
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    make_square(ax)

    return figure
