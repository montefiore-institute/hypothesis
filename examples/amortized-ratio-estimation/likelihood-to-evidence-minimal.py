import argparse
import hypothesis
import matplotlib.pyplot as plt
import torch

from hypothesis.auto.training import LikelihoodToEvidenceRatioEstimatorTrainer as Trainer
from hypothesis.benchmark.normal import Prior
from hypothesis.benchmark.normal import Simulator
from hypothesis.nn.amortized_ratio_estimation import LikelihoodToEvidenceRatioEstimatorMLP as RatioEstimator
from torch.utils.data import TensorDataset



def main(arguments):
    # Allocate the ratio estimator
    estimator = RatioEstimator(
        activation=torch.nn.SELU,
        layers=[128, 128, 128],
        shape_inputs=(1,),
        shape_outputs=(1,))
    estimator = estimator.to(hypothesis.accelerator)
    # Allocate the optimizer
    optimizer = torch.optim.Adam(estimator.parameters())
    # Allocate the trainer, or optimization procedure.
    trainer = Trainer(
        estimator=estimator,
        dataset_train=allocate_dataset_train(),
        dataset_test=allocate_dataset_test(),
        epochs=arguments.epochs,
        checkpoint=arguments.checkpoint,
        batch_size=arguments.batch_size,
        optimizer=optimizer)
    # Execute the optimization process.
    summary = trainer.optimize()


def batch_feeder(batch, criterion, accelerator):
    inputs, outputs = batch
    inputs = inputs.to(accelerator, non_blocking=True)
    outputs = outputs.to(accelerator, non_blocking=True)

    return criterion(inputs=inputs, outputs=outputs)


def allocate_dataset_train():
    return allocate_dataset(1000)


def allocate_dataset_test():
    return allocate_dataset(200)


@torch.no_grad()
def allocate_dataset(n):
    prior = Prior()
    simulator = Simulator()
    size = torch.Size([n])
    inputs = prior.sample(size).view(-1, 1)
    outputs = simulator(inputs).view(-1, 1)

    return TensorDataset(inputs, outputs)


def parse_arguments():
    parser = argparse.ArgumentParser("Amortized Likelihood-to-evidence Ratio Estimation: minimal example")
    parser.add_argument("--batch-size", type=int, default=hypothesis.default.batch_size, help="Batch-size of the stochastic optimization.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to store the checkpoints. If specified, checkpointing will be enabled.")
    parser.add_argument("--epochs", type=int, default=hypothesis.default.epochs, help="Number of data epochs.")
    arguments, _ = parser.parse_known_args()

    return arguments


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
