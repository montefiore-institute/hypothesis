#!/usr/bin/env python

"""Tests tractable benchmark :mod:`hypothesis.benchmark.tractable`.

"""

import hypothesis as h
import pytest
import torch

from hypothesis.benchmark.tractable import Prior
from hypothesis.benchmark.tractable import Simulator


@torch.no_grad()
def test_simulator():
    prior = Prior()
    simulator = Simulator()
    # Test a single sample
    inputs = prior.sample()
    simulator(inputs)
    outputs = simulator.forward(inputs)
    assert outputs.shape[0] == 1
    assert outputs.shape[1] == 8
    # Test multiple samples
    n = 100
    inputs = prior.sample((n,))
    simulator(inputs)
    outputs = simulator.forward(inputs)
    assert outputs.shape[0] == n
    assert outputs.shape[1] == 8
