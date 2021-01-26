#!/usr/bin/env python

"""Tests for the BaseCriterion in :mod:`hypothesis.nn.ratio_estimation.base`.

"""

import hypothesis as h
import numpy as np
import pytest
import time
import torch

from hypothesis.nn.ratio_estimation.mlp import build_ratio_estimator


@torch.no_grad()
def test_criterion_logits():
    # Define the random variables and their shapes (without batch axis).
    batch_size = 4096
    random_variables = {
        "inputs": (10,),
        "outputs": (10,)}

    # Allocate the MLP ratio estimator and the criterion
    r = build_ratio_estimator(random_variables)
    criterion = h.nn.ratio_estimation.BaseCriterion(
        batch_size=batch_size,
        denominator="inputs|outputs",
        estimator=r,
        logits=True)

    # Dummy training loop.
    values = []
    for _ in range(10):
        time_start = time.time()
        for _ in range(100):
            inputs = torch.randn(batch_size, 10)
            outputs = torch.randn(batch_size, 10)
            criterion(inputs=inputs, outputs=outputs)
        delta = time.time() - time_start
        values.append(delta)
    print("\n\nResults, criterion speed with logits:")
    print(" - Mean:", np.mean(values))
    print(" - Std:", np.std(values))


@torch.no_grad()
def test_criterion_no_logits():
    # Define the random variables and their shapes (without batch axis).
    batch_size = 4096
    random_variables = {
        "inputs": (10,),
        "outputs": (10,)}

    # Allocate the MLP ratio estimator and the criterion
    r = build_ratio_estimator(random_variables)
    criterion = h.nn.ratio_estimation.BaseCriterion(
        batch_size=batch_size,
        denominator="inputs|outputs",
        estimator=r,
        logits=False)

    # Dummy training loop.
    values = []
    for _ in range(10):
        time_start = time.time()
        for _ in range(100):
            inputs = torch.randn(batch_size, 10)
            outputs = torch.randn(batch_size, 10)
            criterion(inputs=inputs, outputs=outputs)
        delta = time.time() - time_start
        values.append(delta)
    print("\n\nResults, criterion speed without logits:")
    print(" - Mean:", np.mean(values))
    print(" - Std:", np.std(values))
