"""
Common training utilities.
"""

import hypothesis
import os
import torch
import tqdm

from tqdm import trange



# Global variables.
tbar_epoch = None


def checkpoint_filesystem(model, epoch):
    base = "./models/"
    if not os.path.exists(base):
        os.makedirs(base)
    path = base + str(epoch) + ".th"
    torch.save(model, path)


def initialize_epoch_monitoring(trainer):
    global tbar_epoch

    if tbar_epoch is not None:
        del tbar_epoch
    tbar_epoch = tqdm.trange(trainer.epochs)
    tbar_epoch.set_description("Epochs")


def update_epoch_monitoring(trainer, epoch):
    global tbar_epoch

    tbar_epoch.update()


def register_epoch_monitoring(trainer):
    hypothesis.register_hook(hypothesis.hooks.post_reset, initialize_epoch_monitoring)
    hypothesis.register_hook(hypothesis.hooks.post_epoch, update_epoch_monitoring)
