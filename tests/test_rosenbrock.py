# -*- coding: utf-8 -*-
"""Tests for the Rosenbrock likelihood"""

from unittest.mock import MagicMock, create_autospec, patch

import numpy as np
import pytest

from nessai_models.rosenbrock import (
    Rosenbrock,
    rosenbrock,
    uncoupled_rosenbrock,
)


@pytest.fixture()
def model():
    return create_autospec(Rosenbrock)


@pytest.fixture(params=[rosenbrock, uncoupled_rosenbrock])
def func(request):
    return request.param


@pytest.mark.parametrize("n", [1, 2])
def test_rosenbrock_function(func, n):
    """Assert the output has the correct shape"""
    x = np.random.randn(n, 2)
    out = func(x)
    assert out.ndim == 1
    assert len(out) == n


def test_rosenbrock_function_1d(func):
    """Assert the output has the correct shape for a 1d input"""
    x = np.random.randn(2)
    out = func(x)
    assert out.ndim == 0


def test_rosenbrock_minimum(func):
    """Assert there is minimum at (1, 1).

    This should apply to both versions of the function.
    """
    np.testing.assert_equal(func(np.array([1, 1])), 0.0)


@pytest.mark.parametrize("uncoupled", [False, True])
def test_init(model, uncoupled):
    """Assert the model is instantiated correctly"""
    dims = 4
    bounds = [-1, 1]

    with patch("nessai_models.rosenbrock.NDimensionalModel.__init__") as mock:
        Rosenbrock.__init__(
            model, dims=dims, bounds=bounds, uncoupled=uncoupled
        )

    mock.assert_called_once_with(dims, bounds)

    if uncoupled:
        assert model._fn is uncoupled_rosenbrock
    else:
        assert model._fn is rosenbrock


def test_log_likelihood(model):
    """Assert the correct functions are called."""
    logL = 1.0
    model._fn = MagicMock(return_value=logL)
    model.unstructured_view = MagicMock(return_value="view")
    x = "input"
    out = Rosenbrock.log_likelihood(model, x)

    assert out == -logL
    model.unstructured_view.assert_called_once_with(x)
    model._fn.assert_called_once_with("view")
