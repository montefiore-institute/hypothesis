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
