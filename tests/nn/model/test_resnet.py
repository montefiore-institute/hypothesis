#!/usr/bin/env python

"""Tests for the ResNet models in :mod:`hypothesis.nn.resnet`.

"""

import hypothesis as h
import pytest
import torch


@torch.no_grad()
def test_imports():
    # Define some basic attributes of an MLP.
    shape_xs = (10, 10, 3)
    shape_ys = (10,)
    # Attempt the allocation.
    h.nn.model.resnet.ResNet(shape_xs=shape_xs, shape_ys=shape_ys)
    h.nn.model.ResNet(shape_xs=shape_xs, shape_ys=shape_ys)
    h.nn.ResNet(shape_xs=shape_xs, shape_ys=shape_ys)
