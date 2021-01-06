#!/usr/bin/env python

"""Tests for the ResNet models in :mod:`hypothesis.nn.resnet`.

"""

import hypothesis as h
import pytest
import torch


@torch.no_grad()
def test_imports():
    # Define some basic attributes of an MLP.
    channels = 1
    shape_xs = (50, 50,)
    shape_ys = (10,)
    # Attempt the allocation.
    h.nn.model.resnet.ResNet(shape_xs=shape_xs, shape_ys=shape_ys, channels=channels)
    h.nn.model.ResNet(shape_xs=shape_xs, shape_ys=shape_ys, channels=channels)
    h.nn.ResNet(shape_xs=shape_xs, shape_ys=shape_ys, channels=channels)


@torch.no_grad()
def test_allocation():
    # Define some basic attributes of an MLP.
    shape_xs = (50, 50,)
    shape_ys = (5,)
    # Attempt the allocation
    model = h.nn.ResNet(shape_xs, shape_ys)
    xs = torch.randn(10, 3, 50, 50)
    ys_hat = model(xs)
    assert ys_hat.shape[1] == 5


@torch.no_grad()
def test_normalized_output():
    # Define some basic attributes of an MLP.
    shape_xs = (50, 50,)
    # Sigmoidal normalization
    shape_ys = (1,)
    model = h.nn.ResNet(shape_xs, shape_ys)
    assert isinstance(model._trunk._mapping[-1], torch.nn.Sigmoid)
    # Softmax normalization
    shape_ys = (2,)
    model = h.nn.ResNet(shape_xs, shape_ys)
    assert isinstance(model._trunk._mapping[-1], torch.nn.Softmax)
