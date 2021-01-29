import os
import shutil
import sys
import torch

from hypothesis.util import load_module
from ratio_estimation import compute_coverage
from torch.utils.tensorboard import SummaryWriter


writer = None


def add_hooks(arguments, trainer):
    global writer
    path = arguments.out + "/tensorboard"
    if not os.path.exists(path):
        os.mkdir(path)
    paths = os.listdir(path)
    directories = []
    for p in paths:
        d = path + '/' + p
        if os.path.isdir(d):
            directories.append(p)
    if len(directories) > 0:
        directories.sort()
        last = str(int(directories[-1]) + 1)
    else:
        last = 1
    path += '/' + str(last)
    writer = SummaryWriter(path)
    add_training_monitor(arguments, trainer)
    add_testing_monitor(arguments, trainer)
    add_coverage_monitor(arguments, trainer)


def add_training_monitor(arguments, trainer):
    global writer
    def monitor_train_batch_loss(trainer, batch_index, loss, total_batches, **kwargs):
        if batch_index % 10 == 0:
            identifier = batch_index + trainer.current_epoch * total_batches
            writer.add_scalar("Train batch loss", loss, identifier)
    trainer.add_event_handler(trainer.events.batch_train_complete, monitor_train_batch_loss)
    def monitor_train_loss(trainer, loss, **kwargs):
        writer.add_scalar("Train loss", loss, trainer.current_epoch)
    trainer.add_event_handler(trainer.events.train_complete, monitor_train_loss)


def add_testing_monitor(arguments, trainer):
    global writer
    def monitor_test_batch_loss(trainer, batch_index, loss, total_batches, **kwargs):
        if batch_index % 10 == 0:
            identifier = batch_index + trainer.current_epoch * total_batches
            writer.add_scalar("Test batch loss", loss, identifier)
    trainer.add_event_handler(trainer.events.batch_test_complete, monitor_test_batch_loss)
    def monitor_test_loss(trainer, loss, **kwargs):
        writer.add_scalar("Test loss", loss, trainer.current_epoch)
    trainer.add_event_handler(trainer.events.test_complete, monitor_test_loss)


def add_coverage_monitor(arguments, trainer):
    alpha = 0.05
    if arguments.data_test is not None:
        dataset_size = len(load_module(arguments.data_test)())
        def init_coverage_progress(trainer, **kwargs):
            trainer._init_progress_bottom("Coverage", total=dataset_size)
        def update_coverage(trainer, coverage, **kwargs):
            if trainer._progress_bottom is None:
                return
            trainer._progress_bottom.set_description("Coverage ~ current: {:.4f}".format(coverage))
            trainer._progress_bottom.update()
        @torch.no_grad()
        def coverage(trainer, **kwargs):
            # Check if we have to compute coverage this epoch
            if trainer.current_epoch % 25 != 0:
                return
            init_coverage_progress(trainer, **kwargs)
            # Load the testing dataset and its loader
            dataset = load_module(arguments.data_test)()
            loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=1, num_workers=arguments.workers)
            dataset_size = len(loader)
            covered = 0
            estimator = trainer.best_estimator
            for batch_index, sample_joint in enumerate(loader):
                covered += compute_coverage(
                    alpha=alpha,
                    estimator=estimator,
                    sample_joint=sample_joint)
                current_emperical_coverage = covered / (batch_index + 1)
                update_coverage(trainer, current_emperical_coverage)
            emperical_coverage = covered / dataset_size
            delta = emperical_coverage - (1 - alpha)
            # Change the conservative criterion online through gradient descent.
            trainer.conservativeness = trainer.conservativeness - 0.1 * delta
            # Add to the writer
            writer.add_scalar("Coverage", emperical_coverage, trainer.current_epoch)
            writer.add_scalar("Conservativeness", trainer.conservativeness, trainer.current_epoch)
        # Add the computation of coverage to the trainer.
        trainer.add_event_handler(trainer.events.epoch_complete, coverage)
