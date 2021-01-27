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

from hypothesis.train import RatioEstimatorTrainer as Trainer
from hypothesis.util.data import NamedDataset


def main(arguments):
    # Allocate the datasets
    dataset_test = load_dataset_test(arguments)
    dataset_train = load_dataset_train(arguments)
    dataset_validation = load_dataset_validationc(arguments)
    # Allocate the ratio estimator
    estimator = load_ratio_estimator(arguments)
    # Allocate the optimizer
    optimizer = load_optimizer(arguments, estimator)
    # Allocate the trainer instance
    trainer = Trainer(
        accelerator=h.accelerator,
        batch_size=arguments.batch_size,
        conservativeness=arguments.conservativeness,
        dataset_test=dataset_test,
        dataset_train=dataset_train,
        dataset_validation=dataset_validation,
        epochs=arguments.epochs,
        logits=arguments.logits,
        pin_memory=arguments.pin_memory,
        shuffle=not arguments.dont_shuffle,
        workers=arguments.workers)
    # Add the hooks to the training object.
    add_hooks(arguments, trainer)
    # Start the optimization procedure
    trainer.fit()
    # TODO Save results.


def allocate_dataset_test(arguments):
    if arguments.data_test is not None:
        dataset = load_class(arguments.data_test)()
        assert isinstance(dataset, NamedDataset)
    else:
        dataset = None

    return dataset


def allocate_dataset_train(arguments):
    if arguments.data_train is not None:
        dataset = load_class(arguments.data_train)()
        assert isinstance(dataset, NamedDataset)
    else:
        dataset = None

    return dataset


def allocate_dataset_validate(arguments):
    if arguments.data_validate is not None:
        dataset = load_class(arguments.data_validate)()
        assert isinstance(dataset, NamedDataset)
    else:
        dataset = None

    return dataset


def load_ratio_estimator(arguments):
    RatioEstimator = load_class(arguments.estimator)
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


def add_hooks_display(arguments, trainer):
    pass


def add_hooks_lr_scheduling(arguments, trainer):
    # Check which learning rate scheduler has been specified
    if arguments.lrsched_on_plateau:
        add_hooks_lr_scheduling_on_plateau(arguments, trainer)
    elif arguments.lrsched_cyclic:
        add_hooks_lr_scheduling_cyclic(arguments, trainer)


def add_hooks_lr_scheduling_on_plateau(arguments, trainer):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(trainer.optimizer)
    def schedule(trainer, **kwargs):
        scheduler.step()
    trainer.add_event_handler(trainer.events.epoch_complete, schedule)


def add_hooks_lr_scheduling_cyclic(arguments, trainer):
    scheduler = torch.optim.lr.scheduler.CyclicLR(trainer.optimizer,
        base_lr=arguments.schedlr_cyclic_base_lr,
        max_lr=arguments.schedlr_cyclic_max_lr)
    def schedule(trainer, **kwargs):
        scheduler.step()
    trainer.add_event_handler(trainer.events.batch_train_complete, schedule)


def load_class(full_classname):
    if full_classname is None:
        raise ValueError("The specified classname cannot be `None`.")
    module_name, class_name = full_classname.rsplit('.', 1)
    module = __import__(module_name, fromlist=[class_name])

    return getattr(module, class_name)


def parse_arguments():
    parser = argparse.ArgumentParser()
    # General settings
    parser.add_argument("--data-parallel", action="store_true", help="Enable data-parallel training whenever multiple GPU's are available (default: false).")
    parser.add_argument("--disable-gpu", action="store_true", help="Disable the usage of GPU's (default: false).")
    parser.add_argument("--dont-shuffle", action="store_true", help="Do not shuffle the datasets (default: false).")
    parser.add_argument("--out", type=str, default='.', help="Output directory of the generated files (default: '.').")
    parser.add_argument("--pin-memory", action="store_true", help="Memory map and pipeline data loading to the GPU (default: false).")
    parser.add_argument("--show", action="store_true", help="Show progress of the training to stdout (default: false).")
    # Optimization settings
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size (default: 256).")
    parser.add_argument("--conservativeness", type=float, default=0.0, help="Conservative term (default: 0.0).")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs (default: 1).")
    parser.add_argument("--logits", action="store_true", help="Use the logit-trick for the minimization criterion (default: false).")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001).")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay (default: 0.0).")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent data loaders (default: 4).")
    # Data settings
    parser.add_argument("--data-test", type=str, default=None, help="Full classname of the testing dataset (default: none, optional).")
    parser.add_argument("--data-test", type=str, default=None, help="Full classname of the validation dataset (default: none, optional).")
    parser.add_argument("--data-train", type=str, default=None, help="Full classname of the training dataset (default: none).")
    # Ratio estimator settings
    parser.add_argument("--estimator", type=str, default=None, help="Full classname of the ratio estimator (default: none).")
    # Learning rate scheduling (you can only allocate 1 learning rate scheduler, they will be allocated in the following order.)
    ## Learning rate scheduling on a plateau
    parser.add_argument("--lrsched-on-plateau", action="store_true", help="Enables learning rate scheduling whenever a plateau has been detected (default: false).")
    ## Cyclic learning rate scheduling
    parser.add_argument("--lrsched-cyclic", action="store_true", help="Enables cyclic learning rate scheduling")
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
