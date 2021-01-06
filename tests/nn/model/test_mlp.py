#!/usr/bin/env python

"""Tests for the multi-layered perceptron in :mod:`hypothesis.nn.mlp`.

"""

import hypothesis as h
import pytest
import torch


@torch.no_grad()
def test_imports():
    # Define some basic attributes of an MLP.
    shape_xs = (10,)
    shape_ys = (10,)
    layers = (32, 32, 32)
    # Attempt the allocation.
    h.nn.model.mlp.MLP(shape_xs=shape_xs, shape_ys=shape_ys, layers=layers)
    h.nn.model.MLP(shape_xs=shape_xs, shape_ys=shape_ys, layers=layers)
    h.nn.MLP(shape_xs=shape_xs, shape_ys=shape_ys, layers=layers)


@torch.no_grad()
def test_allocation():
    # Define some basic attributes of an MLP.
    shape_xs = (10,)
    shape_ys = (5,)
    layers = (32, 32, 32)
    # Attempt the allocation
    model = h.nn.MLP(shape_xs, shape_ys, layers=layers)
    xs = torch.randn(10, 10)
    ys_hat = model(xs)
    assert ys_hat.shape[1] == 5


@torch.no_grad()
def test_normalized_output():
    # Define some basic attributes of an MLP.
    shape_xs = (10,)
    layers = (32, 32, 32)
    # Sigmoidal normalization
    shape_ys = (1,)
    model = h.nn.MLP(shape_xs, shape_ys, layers=layers)
    assert isinstance(model._mapping[-1], torch.nn.Sigmoid)
    # Softmax normalization
    shape_ys = (2,)
    model = h.nn.MLP(shape_xs, shape_ys, layers=layers)
    assert isinstance(model._mapping[-1], torch.nn.Softmax)
