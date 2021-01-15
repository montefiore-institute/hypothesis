#!/usr/bin/env python

"""Tests the M/G/1 benchmark :mod:`hypothesis.benchmark.mg1`.

"""

import hypothesis as h
import pytest
import torch

from hypothesis.benchmark.mg1 import Prior
from hypothesis.benchmark.mg1 import Simulator


@torch.no_grad()
def test_simulator():
    prior = Prior()
    simulator = Simulator()
    # Test a single sample
    inputs = prior.sample()
    simulator(inputs)
    outputs = simulator.forward(inputs)
    assert len(outputs.shape) == 2
    assert outputs.shape[0] == 1
    # Test multiple samples
    n = 100
    inputs = prior.sample((n,))
    simulator(inputs)
    outputs = simulator.forward(inputs)
    assert len(outputs.shape) == 2
    assert outputs.shape[0] == n
