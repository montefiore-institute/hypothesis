import hypothesis as h
import hypothesis.workflow as w
import logging
import numpy as np
import os
import torch



@w.root
def initialize():
    logging.info("Starting the simulation-based inference workflow!")


@w.dependency(initialize)
@w.postcondition(w.exists("data/train/inputs.npy"))
@w.postcondition(w.exists("data/train/outputs.npy"))
def simulate_train():
    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists("data/train"):
        os.mkdir("data/train")
    n = 100000
    inputs = np.random.uniform(-15, 15, n)
    outputs = np.random.random(n) + inputs
    logging.info("Training data has been generated.")
    np.save("data/train/inputs.npy", inputs)
    np.save("data/train/outputs.npy", outputs)
    logging.info("Training data has been stored.")


@w.dependency(initialize)
@w.postcondition(w.exists("data/test/inputs.npy"))
@w.postcondition(w.exists("data/test/outputs.npy"))
def simulate_test():
    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists("data/test"):
        os.mkdir("data/test")
    n = 10000
    inputs = np.random.uniform(-15, 15, n)
    outputs = np.random.random(n) + inputs
    logging.info("Testing data has been generated.")
    np.save("data/test/inputs.npy", inputs)
    np.save("data/test/outputs.npy", outputs)
    logging.info("Testing data has been stored.")


@w.dependency(simulate_train)
@w.dependency(simulate_test)
@w.tasks(10)
def train(task):
    logging.critical("No training implementation yet!")
