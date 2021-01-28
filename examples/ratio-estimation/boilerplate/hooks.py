import torch

from ratio_estimation import compute_coverage
from hypothesis.util import load_module


def add_hooks(arguments, trainer):
    add_coverage_hook(arguments, trainer)


def add_coverage_hook(arguments, trainer):
    if arguments.data_test is not None:
        @torch.no_grad()
        def coverage(trainer, **kwargs):
            global p_bottom
            # Check if we have to compute coverage this epoch
            if trainer.current_epoch % 1 != 0:
                return
            # Load the testing dataset and its loader
            dataset = load_module(arguments.data_test)()
            loader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=1, num_workers=arguments.workers)
            dataset_size = len(loader)
            covered = 0
            estimator = trainer.estimator
            # Check if the progress needs to be displayed
            if arguments.show:
                bottom_prefix = "Coverage"
                p_bottom.set_description(bottom_prefix)
                p_bottom.total = dataset_size
                p_bottom.reset()
                p_bottom.refresh()
            for batch_index, sample_joint in enumerate(loader):
                covered += compute_coverage(
                    alpha=arguments.alpha,
                    estimator=estimator,
                    sample_joint=sample_joint)
                # Update the visualization, if requested.
                if arguments.show:
                    current_emperical_coverage = covered / (batch_index + 1)
                    p_bottom.update()
                    p_bottom.set_description("Coverage ~ current: {:.4f}".format(current_emperical_coverage))
            emperical_coverage = covered / dataset_size
            delta = emperical_coverage - (1 - arguments.alpha)
            trainer.conservativeness = trainer.conservativeness - 0.05 * delta
        trainer.add_event_handler(trainer.events.epoch_complete, coverage)
