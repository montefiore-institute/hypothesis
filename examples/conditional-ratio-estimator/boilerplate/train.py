r"""A boilerplate script for training conditional ratio estimators."""

import argparse
import copy
import hypothesis
import numpy as np
import os
import torch

from hypothesis.metric import ExponentialAverageMetric as ExponentialAverage
from hypothesis.nn.conditional_ratio_estimator import ConditionalRatioEstimatorCriterion as Criterion
from hypothesis.util.data.numpy import SimulationDataset as Dataset
from models import DummyRatioEstimator
from models import AnotherDummyRatioEstimator
from torch.utils.data import DataLoader

# Global variables.
best_loss = float("infinity")
best_model = None
best_epoch = 0
dataset_test = None
dataset_train = None
estimator = None
loss_test = []
loss_train = []
lrscheduler = None
optimizer = None



def main(arguments):
    global best_epoch
    global best_model
    global estimator
    global loss_test
    global loss_train
    global lrscheduler

    allocate_dataset_train(arguments)
    allocate_dataset_test(arguments)
    allocate_estimator(arguments)
    allocate_optimizer(arguments)
    allocate_lr_scheduler(arguments)
    last_checkpointed_epoch = load_checkpoint(arguments)
    remaining_epochs = arguments.epochs - last_checkpointed_epoch
    for epoch in range(remaining_epochs):
        train(arguments, epoch)
        test(arguments, epoch)
        lrscheduler.step()
        checkpoint(arguments, epoch)
    # Save the final and best model and loss. Remove the checkpoint.
    torch.save(best_model, arguments.out + "/model.th")
    torch.save(estimator.cpu().state_dict(), arguments.out + "/final_model.th")
    np.save(arguments.out + "/best_epoch.npy", np.array(best_epoch))
    np.save(arguments.out + "/loss_train.npy", np.array(loss_train))
    np.save(arguments.out + "/loss_test.npy", np.array(loss_test))
    os.remove(arguments.out + "/checkpoint")


def train(arguments, epoch):
    global dataset_train
    global estimator
    global loss_train
    global optimizer

    estimator.train()
    criterion = Criterion(estimator, arguments.batch_size)
    criterion.to(hypothesis.accelerator) # Move to GPU, if available.
    loader = allocate_dataloader(arguments, dataset_train)
    device = hypothesis.accelerator
    losses = ExponentialAverage()
    for inputs, outputs in loader:
        inputs = inputs.to(device, non_blocking=True)
        outputs = outputs.to(device, non_blocking=True)
        loss = criterion(inputs, outputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item())
    loss_train.append(losses[-1])


def test(arguments, epoch):
    global best_epoch
    global best_loss
    global best_model
    global dataset_test
    global estimator
    global loss_test

    estimator.eval()
    criterion = Criterion(estimator, arguments.batch_size)
    criterion.to(hypothesis.accelerator) # Move to GPU, if available.
    loader = allocate_dataloader(arguments, dataset_test)
    device = hypothesis.accelerator
    num_batches = len(loader)
    total_loss = 0.0
    for inputs, outputs in loader:
        inputs = inputs.to(device, non_blocking=True)
        outputs = outputs.to(device, non_blocking=True)
        loss = criterion(inputs, outputs)
        total_loss += loss.item()
    total_loss /= num_batches
    loss_test.append(total_loss)
    if total_loss < best_loss:
        # Move to CPU for storage.
        estimator = estimator.cpu()
        best_loss = total_loss
        best_model = estimator.state_dict()
        best_epoch = epoch
        # Move back to GPU.
        estimator = estimator.to(hypothesis.accelerator)


def checkpoint(arguments, epoch):
    global best_epoch
    global best_loss
    global best_model
    global estimator
    global loss_test
    global loss_train
    global lrscheduler
    global optimizer

    state = {
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "best_model": best_model,
        "epoch": epoch,
        "loss_test": loss_test,
        "loss_train": loss_train,
        "lrscheduler": lrscheduler.state_dict(),
        "model": estimator.state_dict(),
        "optimizer": optimizer.state_dict()}
    torch.save(state, arguments.out + "/checkpoint")


def load_checkpoint(arguments):
    global best_epoch
    global best_loss
    global best_model
    global estimator
    global loss_test
    global loss_train
    global lrscheduler
    global optimizer

    # Check if a checkpoint exists.
    path = arguments.out + "/checkpoint"
    if os.path.exists(path) and arguments.load_checkpoint:
        state = torch.load(path)
        last_epoch = state["epoch"]
        loss_test = state["loss_test"]
        loss_train = state["loss_train"]
        best_epoch = state["best_epoch"]
        best_loss = state["best_loss"]
        best_model = state["best_model"]
        lrscheduler.load_state_dict(state["lrscheduler"])
        estimator.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        print("Checkpoint loaded at epoch", last_epoch, ".")
    else:
        last_epoch = 0

    return last_epoch


def allocate_dataset_train(arguments):
    global dataset_train

    dataset_train = None
    # dataset_train = Dataset(
    #     inputs=arguments.inputs_train,
    #     outputs=arguments.outputs_train,
    #     in_memory=True)
    raise NotImplementedError # CHANGEME


def allocate_dataset_test(arguments):
    global dataset_test

    dataset_test = None
    # dataset_train = Dataset(
    #     inputs=arguments.inputs_test,
    #     outputs=arguments.outputs_test,
    #     in_memory=True)
    raise NotImplementedError # CHANGEME


def allocate_estimator(arguments):
    global estimator

    architectures = {
        "dummy": allocate_densenet_estimator,
        "anotherdummy": allocate_mlp_estimator}

    estimator = architectures[arguments.architecture](arguments)
    estimator = estimator.to(hypothesis.accelerator)
    raise NotImplementedError # CHANGEME


def allocate_dummy_estimator(arguments):
    activation = allocate_activation(arguments)

    return DummyRatioEstimator(
        activation=activation,
        dropout=arguments.dropout)


def allocate_anotherdummy_estimator(arguments):
    activation = allocate_activation(arguments)

    return AnotherDummyRatioEstimator(
        activation=activation,
        dropout=arguments.dropout,
        batchnorm=arguments.batchnorm) # Batchnorm can be specified.


def allocate_optimizer(arguments):
    global estimator
    global optimizer

    optimizer = torch.optim.AdamW(estimator.parameters(),
        amsgrad=arguments.amsgrad,
        lr=arguments.lr,
        weight_decay=arguments.weight_decay)


def allocate_lr_scheduler(arguments):
    global optimizer
    global lrscheduler

    lrscheduler = torch.optim.lr_scheduler.StepLR(optimizer,
        gamma=arguments.lrsched_gamma,
        step_size=arguments.lrsched_every)


def allocate_dataloader(arguments, dataset):
     return DataLoader(dataset,
        batch_size=arguments.batch_size,
        drop_last=True,
        pin_memory=arguments.pin_memory,
        num_workers=arguments.workers,
        shuffle=True)


def allocate_activation(arguments):
    activations = {
        "elu": torch.nn.ELU,
        "leakyrelu": torch.nn.LeakyReLU,
        "prelu": torch.nn.PReLU,
        "relu": torch.nn.ReLU,
        "selu": torch.nn.SELU,
        "tanh": torch.nn.Tanh}
    if arguments.activation not in activations.keys():
        raise ValueError("Activation", arguments.activation, "is not available.")

    return activations[arguments.activation]


def parse_arguments():
    parser = argparse.ArgumentParser("Conditional ratio estimator training")
    parser.add_argument("--activation", type=str, default="relu", help="Activation function (default: relu).")
    parser.add_argument("--amsgrad", action="store_true", help="Use AMSGRAD version of Adam (default: false).")
    parser.add_argument("--architecture", type=str, default=None, help="Ratio estimator architecture to train (default: none).")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64).")
    parser.add_argument("--batchnorm", type=int, default=1, help="Enable or disable batch normalization (default: true).")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate (default: 0.0).")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs (default: 1).")
    parser.add_argument("--inputs-test", type=str, default=None, help="Path to the testing inputs (default: none).")
    parser.add_argument("--inputs-train", type=str, default=None, help="Path to the training inputs (default: none).")
    parser.add_argument("--load-checkpoint", action="store_true", help="Load a checkpoint if available (default: false).")
    parser.add_argument("--log-inputs", action="store_true", help="Trains with inputs in log (ln) scale (default: false).")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001).")
    parser.add_argument("--lrsched-every", type=int, default=1000, help="Schedule the learning rate every n epochs (default: 100).")
    parser.add_argument("--lrsched-gamma", type=float, default=0.75, help="Decay factor of the learning rate scheduler (default: 0.75).")
    parser.add_argument("--out", type=str, default=None, help="Output directory fro the model.")
    parser.add_argument("--outputs-test", type=str, default=None, help="Path to the testing outputs (default: none).")
    parser.add_argument("--outputs-train", type=str, default=None, help="Path to the training outputs (default: none).")
    parser.add_argument("--pin-memory", action="store_true", help="Memory-map the data of the data loader (default: false).")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay (default: 0.0).")
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent data loaders (default: 1).")
    arguments, _ = parser.parse_known_args()
    arguments.batchnorm = (arguments.batchnorm != 0)

    return arguments


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
