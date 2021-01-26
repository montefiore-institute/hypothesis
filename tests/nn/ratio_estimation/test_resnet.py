#!/usr/bin/env python

"""Tests for :mod:`hypothesis.nn.ratio_estimation.resnet`.

"""

import hypothesis as h
import numpy as np
import pytest
import time
import torch

from hypothesis.nn.ratio_estimation.resnet import build_ratio_estimator


@torch.no_grad()
def test_allocation_resnet():
    random_variables = {
        "inputs": (15,),
        "outputs": (3, 15, 15)}  # Channel x Width x Height
    r = build_ratio_estimator(random_variables)()
    assert r is not None
