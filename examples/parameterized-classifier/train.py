import argparse
import hypothesis
import os
import torch
import tqdm

from hypothesis.benchmark.normal import NormalSimulator as Simulator
from hypothesis.io.dataset import GeneratorDataset
from hypothesis.nn import ParameterizedClassifier
from hypothesis.train import ParameterizedClassifierTrainer
from torch.distributions.uniform import Uniform



# Global variables.
progress_bar = None

def main(arguments):
    # Allocate the checkpointing lambda.
    def checkpoint_filesystem(model, epoch):
        base = arguments.output + '/'
        if not os.path.exists(base):
            os.makedirs(base)
        path = base + str(epoch)
        torch.save(model, path)
    # Allocate the training utilities.
    dataset = allocate_dataset(arguments)
    classifier = allocate_classifier(arguments)
    trainer = ParameterizedClassifierTrainer(dataset, allocate_optimizer,
        epochs=arguments.epochs, data_workers=arguments.data_workers,
        batch_size=arguments.batch_size, checkpoint=checkpoint_filesystem,
        allocate_scheduler=allocate_scheduler)
    hypothesis.clear_hooks()
    # Check if progress needs to be displayed to stdout.
    if arguments.show_progress:
        hypothesis.register_hook(hypothesis.hooks.post_reset, hook_before_start)
        hypothesis.register_hook(hypothesis.hooks.post_epoch, hook_post_epoch)
        hypothesis.register_hook(hypothesis.hooks.end, hook_end)
    trainer.train(classifier)
    print("Training completed!")


def hook_before_start(trainer):
    global progress_bar
    progress_bar = tqdm.tqdm(total=trainer.epochs)


def hook_post_epoch(trainer, epoch):
    global progress_bar
    progress_bar.update(1)


def hook_end(trainer):
    global progress_bar
    progress_bar.close()


def allocate_optimizer(classifier):
    return torch.optim.Adam(classifier.parameters())


def allocate_scheduler(optimizer):
    return torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=.7)


def allocate_dataset(arguments):
    prior = Uniform(arguments.lower, arguments.upper)
    simulator = Simulator()
    dataset = GeneratorDataset(simulator, prior)

    return dataset


def allocate_classifier(arguments):
    hidden = arguments.hidden
    classifier = torch.nn.Sequential(
        torch.nn.Linear(2, hidden),
        torch.nn.SELU(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.SELU(),
        torch.nn.Linear(hidden, 1),
        torch.nn.Sigmoid())
    classifier = ParameterizedClassifier(
        classifier, lower=arguments.lower, upper=arguments.upper)

    return classifier


def parse_arguments():
    parser = argparse.ArgumentParser("Univariate normal - Training.")
    parser.add_argument("--hidden", type=int, default=64, help="Number of hidden units of the classifier.")
    parser.add_argument("--lower", type=float, default=-10, help="Lower bound of the search space.")
    parser.add_argument("--upper", type=float, default=10, help="Upper bound of the search space.")
    parser.add_argument("--output", type=str, default=None, help="Output directory.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch-size of the optimization procedure.")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of data iterations.")
    parser.add_argument("--data-workers", type=int, default=0, help="Number of asynchronous data loaders.")
    parser.add_argument("--show-progress", type=bool, default=False, nargs='?', const=True, help="Shows the progress of the training.")
    arguments, _ = parser.parse_known_args()
    arguments.lower = torch.tensor([arguments.lower]).float()
    arguments.upper = torch.tensor([arguments.upper]).float()
    # Check if an output directory has been specified.
    if arguments.output is None:
        raise Exception("No output directory has been specified.")

    return arguments


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
