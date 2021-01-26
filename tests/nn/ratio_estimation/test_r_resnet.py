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
        "outputs": (3, 64, 64)}  # Channel x Width x Height
    # By default the 'outputs' variable is convolved.
    r = build_ratio_estimator(random_variables)()
    assert r is not None

    batch_size = 10
    inputs = torch.randn(batch_size, 15)
    outputs = torch.randn(batch_size, 3, 64, 64)
    log_ratios = r.log_ratio(inputs=inputs, outputs=outputs)
    assert log_ratios.shape[0] == batch_size
    assert log_ratios.shape[1] == 1

    batch_size = 1
    inputs = torch.randn(batch_size, 15)
    outputs = torch.randn(batch_size, 3, 64, 64)
    log_ratios = r.log_ratio(inputs=inputs, outputs=outputs)
    assert log_ratios.shape[0] == batch_size
    assert log_ratios.shape[1] == 1


@torch.no_grad()
def test_multi_head_resnet():
    random_variables = {
        "inputs": (15,),
        "outputs1": (3, 64, 64),  # Channel x Width x Height (2D)
        "outputs2": (1, 48, 48),  # Channel x Width x Height (2D)
        "outputs3": (1, 512)}     # Channel x Width (1D)
    convolve_variables = [
        "outputs1", "outputs2", "outputs3"]
    r = build_ratio_estimator(
        random_variables,
        convolve=convolve_variables)()
    assert r is not None

    batch_size = 10
    inputs = torch.randn(batch_size, 15)
    outputs1 = torch.randn(batch_size, 3, 64, 64)
    outputs2 = torch.randn(batch_size, 1, 48, 48)
    outputs3 = torch.randn(batch_size, 1, 512)
    log_ratios = r.log_ratio(
        inputs=inputs,
        outputs1=outputs1,
        outputs2=outputs2,
        outputs3=outputs3)
    assert log_ratios.shape[0] == batch_size
    assert log_ratios.shape[1] == 1

    batch_size = 1
    inputs = torch.randn(batch_size, 15)
    outputs1 = torch.randn(batch_size, 3, 64, 64)
    outputs2 = torch.randn(batch_size, 1, 48, 48)
    outputs3 = torch.randn(batch_size, 1, 512)
    log_ratios = r.log_ratio(
        inputs=inputs,
        outputs1=outputs1,
        outputs2=outputs2,
        outputs3=outputs3)
    assert log_ratios.shape[0] == batch_size
    assert log_ratios.shape[1] == 1


@torch.no_grad()
def test_multi_head_multi_depth_resnet():
    random_variables = {
        "inputs": (15,),
        "outputs1": (3, 64, 64),  # Channel x Width x Height (2D)
        "outputs2": (1, 48, 48),  # Channel x Width x Height (2D)
        "outputs3": (1, 512)}     # Channel x Width (1D)
    convolve_variables = [
        "outputs1", "outputs2", "outputs3"]
    r = build_ratio_estimator(
        random_variables,
        depth=[18, 34, 50],
        convolve=convolve_variables)()
    assert r is not None

    batch_size = 10
    inputs = torch.randn(batch_size, 15)
    outputs1 = torch.randn(batch_size, 3, 64, 64)
    outputs2 = torch.randn(batch_size, 1, 48, 48)
    outputs3 = torch.randn(batch_size, 1, 512)
    log_ratios = r.log_ratio(
        inputs=inputs,
        outputs1=outputs1,
        outputs2=outputs2,
        outputs3=outputs3)
    assert log_ratios.shape[0] == batch_size
    assert log_ratios.shape[1] == 1

    batch_size = 1
    inputs = torch.randn(batch_size, 15)
    outputs1 = torch.randn(batch_size, 3, 64, 64)
    outputs2 = torch.randn(batch_size, 1, 48, 48)
    outputs3 = torch.randn(batch_size, 1, 512)
    log_ratios = r.log_ratio(
        inputs=inputs,
        outputs1=outputs1,
        outputs2=outputs2,
        outputs3=outputs3)
    assert log_ratios.shape[0] == batch_size
    assert log_ratios.shape[1] == 1
