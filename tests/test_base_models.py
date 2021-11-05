# -*- coding: utf-8 -*-
"""Tests the base models from `nessai_models.base`."""
import numpy as np
import pytest
from unittest.mock import MagicMock, create_autospec

from nessai_models.base import (
    NDimensionalModel,
    UniformPriorMixin,
)


@pytest.mark.parametrize('bounds', [[-10.0, 10.0], np.array([-10.0, 10.0])])
def test_n_dimensional_model_bounds(bounds):
    """Test the n-dimensional model init."""
    model = create_autospec(NDimensionalModel)
    NDimensionalModel.__init__(model, 2, [-10, 10])
    assert model.names == ['x_0', 'x_1']
    np.testing.assert_equal(model.bounds['x_0'], [-10.0, 10.0])
    np.testing.assert_equal(model.bounds['x_1'], [-10.0, 10.0])


def test_n_dimensional_model_bounds_invalid_type():
    """Assert an error is raised in the bounds are the incorrect type."""
    model = create_autospec(NDimensionalModel)
    with pytest.raises(TypeError) as excinfo:
        NDimensionalModel.__init__(model, 2, 10)
    assert 'Invalid type' in str(excinfo.value)


def test_n_dimensional_model_bounds_invalid_length():
    """Assert an error is raised in the bounds are the incorrect length"""
    model = create_autospec(NDimensionalModel)
    with pytest.raises(ValueError) as excinfo:
        NDimensionalModel.__init__(model, 2, [1, 2, 3])
    assert 'must have length 2' in str(excinfo.value)


def test_uniform_prior_mixin():
    """Assert the value returned by log-prior method is correct."""
    # Values won't be checked because of mocked method in_bounds
    x = np.zeros(10)
    lower_bounds = np.array([-10, -2, 2])
    upper_bounds = np.array([10, 1, 7])
    target = -np.log(20) - np.log(3) - np.log(5)
    obj = create_autospec(UniformPriorMixin)
    obj.in_bounds = MagicMock(return_value=np.ones(x.size).astype(bool))
    obj.lower_bounds = lower_bounds
    obj.upper_bounds = upper_bounds

    log_prob = UniformPriorMixin.log_prior(obj, x)

    obj.in_bounds.assert_called_once_with(x)
    np.testing.assert_equal(log_prob, target)


def test_uniform_prior_mixin_out_of_bounds():
    """Test the log-prior method when a point is deemed out of bounds"""
    # Values won't be checked because of mocked method in_bounds
    x = np.zeros(2)
    lower_bounds = np.array([-10, -2, 2])
    upper_bounds = np.array([10, 1, 7])
    target = np.array([-np.log(20) - np.log(3) - np.log(5), -np.inf])
    obj = create_autospec(UniformPriorMixin)
    obj.in_bounds = MagicMock(return_value=np.array([True, False]))
    obj.lower_bounds = lower_bounds
    obj.upper_bounds = upper_bounds

    log_prob = UniformPriorMixin.log_prior(obj, x)

    obj.in_bounds.assert_called_once_with(x)
    np.testing.assert_equal(log_prob, target)
