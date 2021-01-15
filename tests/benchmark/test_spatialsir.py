#!/usr/bin/env python

"""Tests the Spatial SIR benchmark :mod:`hypothesis.benchmark.spatialsir`.

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
    simulator(inputs)
    outputs = simulator.forward(inputs)
    assert len(outputs.shape) == 4
    assert outputs.shape[0] == 1
    # Test multiple samples
    n = 100
    inputs = prior.sample((n,))
    simulator(inputs)
    outputs = simulator.forward(inputs)
    assert len(outputs.shape) == 4
    assert outputs.shape[0] == n


@torch.no_grad()
def test_consistency():
    prior = Prior()
    simulator = Simulator()
    n = 1000
    for _ in range(n):
        sample = prior.sample()
        outputs = simulator(sample)
        assert (outputs[0, 0, :, :] | outputs[0, 1, :, :] | outputs[0, 2, :, :]).sum() == 100 ** 2
