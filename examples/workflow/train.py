import hypothesis as h
import hypothesis.workflow as w
import logging
import numpy as np
import os
import torch

from hypothesis.nn.ratio_estimation import build_mlp_estimator
from hypothesis.train import RatioEstimatorTrainer as Trainer
from hypothesis.util.data import NamedDataset
from hypothesis.util.data import NumpyDataset


@w.root
@w.slurm.name("JOBINIT")
def initialize():
    logging.info("Starting the simulation-based inference workflow!")
    # Create the data directories
    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists("data/train"):
        os.mkdir("data/train")
    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists("data/test"):
        os.mkdir("data/test")


@w.dependency(initialize)
@w.postcondition(w.exists("data/train/inputs.npy"))
@w.postcondition(w.exists("data/train/outputs.npy"))
@w.slurm.name("SIMULATE_TRAIN")
@w.slurm.cpu_and_memory(1, "256M")
@w.slurm.timelimit("00:10:00")
def simulate_train():
    n = 100000
    inputs = np.random.uniform(-15, 15, n)
    outputs = np.random.random(n) + inputs
    inputs = inputs.reshape(-1, 1)
    outputs = outputs.reshape(-1, 1)
    logging.info("Training data has been generated.")
    np.save("data/train/inputs.npy", inputs.astype(np.float32))
    np.save("data/train/outputs.npy", outputs.astype(np.float32))
    logging.info("Training data has been stored.")


@w.dependency(initialize)
@w.postcondition(w.exists("data/test/inputs.npy"))
@w.postcondition(w.exists("data/test/outputs.npy"))
@w.slurm.name("SIMULATE_TEST")
@w.slurm.cpu_and_memory(1, "256M")
@w.slurm.timelimit("00:10:00")
def simulate_test():
    n = 10000
    inputs = np.random.uniform(-15, 15, n)
    outputs = np.random.random(n) + inputs
    inputs = inputs.reshape(-1, 1)
    outputs = outputs.reshape(-1, 1)
    logging.info("Testing data has been generated.")
    np.save("data/test/inputs.npy", inputs.astype(np.float32))
    np.save("data/test/outputs.npy", outputs.astype(np.float32))
    logging.info("Testing data has been stored.")


@torch.no_grad()
def epoch_handler(trainer, epoch, **kwargs):
    logging.warning("The current epoch is: " + str(epoch))


@torch.no_grad()
def batch_handler(trainer, batch_index, loss, **kwargs):
    if batch_index % 100 == 0:
        logging.error(" - Current loss: " + str(loss.item()))


@w.dependency(simulate_train)
@w.dependency(simulate_test)
@w.tasks(10)
@w.slurm.name("TRAIN")
@w.slurm.cpu_and_memory(6, "4G")
@w.slurm.timelimit("00:10:00")
@w.slurm.gpu(1)
def train(task):
    logging.info("Executing training task " + str(task))

    # Training dataset
    logging.info("Loading the training dataset")
    dataset_train_inputs = NumpyDataset("data/train/inputs.npy")
    dataset_train_outputs = NumpyDataset("data/train/outputs.npy")
    dataset_train = NamedDataset(
        inputs=dataset_train_inputs,
        outputs=dataset_train_outputs)

    # Testing dataset
    logging.info("Loading the testing dataset")
    dataset_test_inputs = NumpyDataset("data/test/inputs.npy")
    dataset_test_outputs = NumpyDataset("data/test/outputs.npy")
    dataset_test = NamedDataset(
        inputs=dataset_test_inputs,
        outputs=dataset_test_outputs)

    # Allocate the model and the optimizer
    random_variables = {"inputs": (1,), "outputs": (1,)}
    RatioEstimator = build_mlp_estimator(random_variables)
    r = RatioEstimator()  # Allocate the generated class
    r = r.to(h.accelerator)
    optimizer = torch.optim.Adam(r.parameters())

    # Allocate the trainer
    trainer = Trainer(
        batch_size=128,
        dataset_test=dataset_test,
        dataset_train=dataset_train,
        epochs=5,
        estimator=r,
        optimizer=optimizer)

    # Append the hooks to the trainer
    trainer.add_event_handler(trainer.events.epoch_start, epoch_handler)
    trainer.add_event_handler(trainer.events.batch_train_complete, batch_handler)

    # Run the training procedure
    trainer.fit()
