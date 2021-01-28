r"""Utility program to train ratio estimators.

This program provides a whole range of utilities to
monitor and train ratio estimators in various ways.
All defined through command line arguments!

"""

import argparse
import hypothesis as h
import hypothesis.workflow as w
import numpy as np
import os
import torch

from hypothesis.train import RatioEstimatorTrainer as Trainer
from hypothesis.util import load_module
from hypothesis.util.data import NamedDataset
from tqdm import tqdm


# Globals
p_bottom = None
p_top = None


def main(arguments):
    # Allocate the datasets
    dataset_test = load_dataset_test(arguments)
    dataset_train = load_dataset_train(arguments)
    dataset_validate = load_dataset_validate(arguments)
    # Allocate the ratio estimator
    estimator = load_ratio_estimator(arguments)
    # Allocate the optimizer
    optimizer = load_optimizer(arguments, estimator)
    # Allocate the trainer instance
    trainer = Trainer(
        estimator=estimator,
        optimizer=optimizer,
        accelerator=h.accelerator,
        batch_size=arguments.batch_size,
        conservativeness=arguments.conservativeness,
        dataset_test=dataset_test,
        dataset_train=dataset_train,
        dataset_validate=dataset_validate,
        epochs=arguments.epochs,
        logits=arguments.logits,
        pin_memory=arguments.pin_memory,
        shuffle=not arguments.dont_shuffle,
        workers=arguments.workers)
    # Add the hooks to the training object.
    add_hooks(arguments, trainer)
    # Start the optimization procedure
    trainer.fit()
    # Save the generated results.
    if os.path.isdir(arguments.out):
        # Save the associated losses.
        if len(trainer.losses_test) > 0:
            np.save(arguments.out + "/losses-test.npy", trainer.losses_test)
        if len(trainer.losses_validate) > 0:
            np.save(arguments.out + "/losses-validation.npy", trainer.losses_validate)
        if len(trainer.losses_train) > 0:
            np.save(arguments.out + "/losses-train.npy", trainer.losses_train)
        # Save the state dict of the best ratio estimator
        torch.save(trainer.best_state_dict, arguments.out + "/weights.th")
        torch.save(trainer.state_dict, arguments.out + "/weights-final.th")
        # Check if coverages have been computed
        if len(coverages) > 0:
            confidence_level = 1 - arguments.alpha
            np.save(arguments.out + "/coverages-" + str(confidence_level) + ".npy", coverages)


def load_dataset_test(arguments):
    if arguments.data_test is not None:
        dataset = load_module(arguments.data_test)()
        assert isinstance(dataset, NamedDataset)
    else:
        dataset = None

    return dataset


def load_dataset_train(arguments):
    if arguments.data_train is not None:
        dataset = load_module(arguments.data_train)()
        assert isinstance(dataset, NamedDataset)
    else:
        dataset = None

    return dataset


def load_dataset_validate(arguments):
    if arguments.data_validate is not None:
        dataset = load_module(arguments.data_validate)()
        assert isinstance(dataset, NamedDataset)
    else:
        dataset = None

    return dataset


def load_ratio_estimator(arguments):
    RatioEstimator = load_module(arguments.estimator)
    estimator = RatioEstimator()
    # Check if we are able to allocate a data parallel model.
    if torch.cuda.device_count() > 1 and arguments.data_parallel:
        estimator = torch.nn.DataParallel(estimator)
    estimator = estimator.to(h.accelerator)

    return estimator


def load_optimizer(arguments, estimator):
    optimizer = torch.optim.AdamW(
        estimator.parameters(),
        lr=arguments.lr,
        weight_decay=arguments.weight_decay)

    return optimizer


def add_hooks(arguments, trainer):
    # Add the display hooks
    add_hooks_display(arguments, trainer)
    # Add the learning rate scheduling hooks
    add_hooks_lr_scheduling(arguments, trainer)
    # Check if a custom hook method has been specified.
    if arguments.hooks is not None:
        hook_loader = load_module(arguments.hooks)
        hook_loader(arguments, trainer)


@torch.no_grad()
def add_hooks_display(arguments, trainer):
    global p_top
    global p_bottom
    r"""Epochs represent the top line, batches the bottom."""
    # Check if the progress needs to be shown to stdout.
    if not arguments.show:
        return
    # Define the progress bars.
    top_prefix = "Epochs"
    bottom_prefix = "Training"
    p_top = tqdm(total=arguments.epochs, desc=top_prefix)
    p_bottom = tqdm()
    # Define the hooks.
    def start_training(trainer, **kwargs):
        global bottom_prefix
        global p_bottom
        bottom_prefix = "Training"
        p_bottom.set_description(bottom_prefix)
        p_bottom.total = None
        p_bottom.reset()
        p_bottom.refresh()
    def start_testing(trainer, **kwargs):
        global bottom_prefix
        global p_bottom
        bottom_prefix = "Testing"
        p_bottom.set_description(bottom_prefix)
        p_bottom.total = None
        p_bottom.reset()
        p_bottom.refresh()
    def start_validation(trainer, **kwargs):
        global bottom_prefix
        global p_bottom
        bottom_prefix = "Validation"
        p_bottom.set_description(bottom_prefix)
        p_bottom.total = None
        p_bottom.reset()
        p_bottom.refresh()
    def update_batch(trainer, loss, batch_index, total_batches, **kwargs):
        global bottom_prefix
        if p_bottom.total is None:
            p_bottom.total = total_batches
            p_bottom.set_description(bottom_prefix + " ~ current loss {:.4f}".format(loss))
            p_bottom.refresh()
        if batch_index % 10 == 0:
            p_bottom.set_description(bottom_prefix + " ~ current loss: {:.4f}".format(loss))
        p_bottom.update()
    def update_epoch(trainer, **kwargs):
        epoch = trainer.current_epoch
        if len(trainer.losses_test) > 0:
            best_loss = np.min(trainer.losses_test)
            p_top.set_description(top_prefix + " ~ best test loss: {:.4f}".format(best_loss))
        p_top.update()
    # Register the hooks
    trainer.add_event_handler(trainer.events.train_start, start_training)
    trainer.add_event_handler(trainer.events.test_start, start_testing)
    trainer.add_event_handler(trainer.events.validate_start, start_validation)
    trainer.add_event_handler(trainer.events.batch_test_complete, update_batch)
    trainer.add_event_handler(trainer.events.batch_train_complete, update_batch)
    trainer.add_event_handler(trainer.events.batch_validate_complete, update_batch)
    trainer.add_event_handler(trainer.events.epoch_complete, update_epoch)


def add_hooks_lr_scheduling(arguments, trainer):
    # Check which learning rate scheduler has been specified
    if arguments.lrsched_on_plateau:
        add_hooks_lr_scheduling_on_plateau(arguments, trainer)
    elif arguments.lrsched_cyclic:
        add_hooks_lr_scheduling_cyclic(arguments, trainer)


def add_hooks_lr_scheduling_on_plateau(arguments, trainer):
    # Check if a test set is available, as the scheduler required a metric.
    if arguments.data_test is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(trainer.optimizer)
        def schedule(trainer, **kwargs):
            scheduler.step(trainer.losses_test[-1])
        trainer.add_event_handler(trainer.events.epoch_complete, schedule)


def add_hooks_lr_scheduling_cyclic(arguments, trainer):
    scheduler = torch.optim.lr_scheduler.CyclicLR(trainer.optimizer,
        cycle_momentum=False,
        base_lr=arguments.lrsched_cyclic_base_lr,
        max_lr=arguments.lrsched_cyclic_max_lr)
    def schedule(trainer, **kwargs):
        scheduler.step()
    trainer.add_event_handler(trainer.events.batch_train_complete, schedule)


def parse_arguments():
    parser = argparse.ArgumentParser()
    # General settings
    parser.add_argument("--data-parallel", action="store_true", help="Enable data-parallel training whenever multiple GPU's are available (default: false).")
    parser.add_argument("--disable-gpu", action="store_true", help="Disable the usage of GPU's (default: false).")
    parser.add_argument("--dont-shuffle", action="store_true", help="Do not shuffle the datasets (default: false).")
    parser.add_argument("--hooks", type=str, default=None, help="Method name (including module) to which adds custom hooks to the trainer (default: none).")
    parser.add_argument("--out", type=str, default='.', help="Output directory of the generated files (default: '.').")
    parser.add_argument("--pin-memory", action="store_true", help="Memory map and pipeline data loading to the GPU (default: false).")
    parser.add_argument("--show", action="store_true", help="Show progress of the training to stdout (default: false).")
    # Optimization settings
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size (default: 256).")
    parser.add_argument("--conservativeness", type=float, default=0.0, help="Conservative term (default: 0.0).")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs (default: 1).")
    parser.add_argument("--logits", action="store_true", help="Use the logit-trick for the minimization criterion (default: false).")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate (default: 0.001).")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay (default: 0.0).")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent data loaders (default: 4).")
    # Data settings
    parser.add_argument("--data-test", type=str, default=None, help="Full classname of the testing dataset (default: none, optional).")
    parser.add_argument("--data-train", type=str, default=None, help="Full classname of the training dataset (default: none).")
    parser.add_argument("--data-validate", type=str, default=None, help="Full classname of the validation dataset (default: none, optional).")
    # Ratio estimator settings
    parser.add_argument("--estimator", type=str, default=None, help="Full classname of the ratio estimator (default: none).")
    # Learning rate scheduling (you can only allocate 1 learning rate scheduler, they will be allocated in the following order.)
    ## Learning rate scheduling on a plateau
    parser.add_argument("--lrsched-on-plateau", action="store_true", help="Enables learning rate scheduling whenever a plateau has been detected (default: false).")
    ## Cyclic learning rate scheduling
    parser.add_argument("--lrsched-cyclic", action="store_true", help="Enables cyclic learning rate scheduling. Requires a test dataset to be specified (default: true).")
    parser.add_argument("--lrsched-cyclic-base-lr", type=float, default=None, help="Base learning rate of the scheduler (default: --lr / 10).")
    parser.add_argument("--lrsched-cyclic-max-lr", type=float, default=None, help="Maximum learning rate of the scheduler (default: --lr).")
    # Parse the supplied arguments
    arguments, _ = parser.parse_known_args()

    # Set the default options of the cyclic learning rate scheduler
    if arguments.lrsched_cyclic_base_lr is None:
        arguments.lrsched_cyclic_base_lr = arguments.lr / 10
    if arguments.lrsched_cyclic_max_lr is None:
        arguments.lrsched_cyclic_max_lr = arguments.lr

    return arguments


if __name__ == "__main__":
    arguments = parse_arguments()
    if arguments.disable_gpu:
        h.disable_gpu()
    main(arguments)
