#!/usr/bin/env python

"""Tests for :mod:`hypothesis.train.ratio_estimation`.

"""

import hypothesis as h
import numpy as np
import pytest
import time
import torch

from hypothesis.nn.ratio_estimation.mlp import build_ratio_estimator
from hypothesis.train import RatioEstimatorTrainer as Trainer
from hypothesis.util.data import NamedDataset


def allocate_dataset(size):
    return torch.utils.data.TensorDataset(torch.randn(size, 15))


@torch.no_grad()
def test_basic_trainer():
    # Allocate the training dataset
    n_train = 1000
    dataset_train_inputs = allocate_dataset(n_train)
    dataset_train_outputs = allocate_dataset(n_train)
    dataset_train = NamedDataset(inputs=dataset_train_inputs, outputs=dataset_train_outputs)
    # Allocate the training dataset
    n_test = 100
    dataset_test_inputs = allocate_dataset(n_test)
    dataset_test_outputs = allocate_dataset(n_test)
    dataset_test = NamedDataset(inputs=dataset_test_inputs, outputs=dataset_test_outputs)
    # Allocate the ratio estimator
    random_variables = {
        "inputs": (15,),
        "outputs": (15,)}
    r = build_ratio_estimator(random_variables)()
    r = r.to(h.accelerator)
    # Allocate the training and the optimizer
    batch_size = 32
    optimizer = torch.optim.Adam(r.parameters())
    trainer = Trainer(
        accelerator=h.accelerator,
        batch_size=batch_size,
        dataset_test=dataset_test,
        dataset_train=dataset_train,
        epochs=100,
        estimator=r,
        optimizer=r)
    trainer.fit()
    assert r is not None
    assert trainer.current_epoch == 99
