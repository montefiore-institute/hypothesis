#!/usr/bin/env python

"""Tests Spatial SIR benchmark :mod:`hypothesis.benchmark.spatialsir`.

"""

import hypothesis as h
import pytest
import torch

from hypothesis.benchmark.spatialsir import Prior
from hypothesis.benchmark.spatialsir import PriorExperiment
from hypothesis.benchmark.spatialsir import Simulator


@torch.no_grad()
def test_simulator():
    prior = Prior()
    simulator = Simulator()
    # Test a single sample
    inputs = prior.sample()
    print(inputs)
    simulator(inputs)
    outputs = simulator.forward(inputs)
    print(outputs.shape)
    assert len(outputs.shape) == 4
    assert outputs.shape[0] == 1
    # Test multiple samples
    n = 100
    inputs = prior.sample((n,))
    simulator(inputs)
    outputs = simulator.forward(inputs)
    assert len(outputs.shape) == 4
    assert outputs.shape[0] == n
