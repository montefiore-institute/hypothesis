#!/usr/bin/env python

"""Tests tractable benchmark :mod:`hypothesis.benchmark.tractable`.

"""

import hypothesis as h
import pytest
import torch

from hypothesis.benchmark.weinberg import Prior
from hypothesis.benchmark.weinberg import PriorExperiment
from hypothesis.benchmark.weinberg import Simulator


@torch.no_grad()
def test_simulator():
    prior = Prior()
    simulator = Simulator()
    # Test a single sample
    inputs = prior.sample()
    simulator(inputs)
    outputs = simulator.forward(inputs)
    assert outputs.shape[0] == 1
    assert outputs.shape[1] == 1
    # Test multiple samples
    n = 100
    inputs = prior.sample((n,))
    simulator(inputs)
    outputs = simulator.forward(inputs)
    assert outputs.shape[0] == n
    assert outputs.shape[1] == 1


@torch.no_grad()
def test_low_beam_energy():
    repeat = 10
    for _ in range(repeat):
        n = 10000
        prior = Prior()
        configurations = torch.tensor(n * [40.0])
        inputs = prior.sample((n,))
        simulator = Simulator()
        outputs = simulator(inputs, configurations)
        assert outputs.mean() < 0  # The mean should definitly be smaller than 0


@torch.no_grad()
def test_insensitive_beam_energy():
    repeat = 10
    for _ in range(repeat):
        n = 10000
        prior = Prior()
        configurations = torch.tensor(n * [45.0])
        inputs = prior.sample((n,))
        simulator = Simulator()
        outputs = simulator(inputs, configurations)
        assert outputs.mean().abs() < 0.05  # The mean should approximately be 0.


@torch.no_grad()
def test_high_beam_energy():
    repeat = 10
    for _ in range(repeat):
        n = 10000
        prior = Prior()
        configurations = torch.tensor(n * [50.0])
        inputs = prior.sample((n,))
        simulator = Simulator()
        outputs = simulator(inputs, configurations)
        assert outputs.mean() > 0  # The mean should definitly be larger than 0
