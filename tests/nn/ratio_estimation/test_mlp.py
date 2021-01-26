#!/usr/bin/env python

"""Tests for :mod:`hypothesis.nn.ratio_estimation.mlp`.

"""

import hypothesis as h
import numpy as np
import pytest
import time
import torch

from hypothesis.nn.ratio_estimation.mlp import build_ratio_estimator


@torch.no_grad()
def test_allocation_mlp():
    random_variables = {
        "inputs": (15,),
        "outputs": (15,)}
    r = build_ratio_estimator(random_variables)()
    assert r is not None
