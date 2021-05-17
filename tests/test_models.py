# -*- coding: utf-8 -*-
"""
Generic tests for all of the models.
"""
import numpy as np
import pytest
from unittest.mock import create_autospec

from nessai.livepoint import numpy_array_to_live_points

from nessaimodels import (
    Gaussian,
    Rosenbrock
)

models = [Gaussian, Rosenbrock]


@pytest.fixture(params=models)
def Model(request):
    return request.param


@pytest.mark.parametrize('dims', [2, 32])
@pytest.mark.parametrize('bounds', [(-1, 1), (-10, 10)])
def test_init(Model, dims, bounds):
    """Test the init method"""
    model = Model(n=dims, bounds=bounds)
    assert model.dims == dims


@pytest.mark.parametrize('dims', [2, 32])
@pytest.mark.parametrize('bounds', [(-1, 1), (-10, 10)])
@pytest.mark.parametrize('n', [1, 10, 100])
def test_log_prior(Model, dims, bounds, n):
    """Test the log-prior method"""
    model = create_autospec(Model)
    model.names = [f'x_{i}' for i in range(dims)]
    model.bounds = {name: bounds for name in model.names}
    x = numpy_array_to_live_points(
        np.random.uniform(low=bounds[0], high=bounds[1], size=(n, dims)),
        model.names
    )
    log_p = Model.log_prior(model, x)
    assert log_p.ndim == 1
    assert len(log_p) == n
    assert np.isfinite(log_p).all()


@pytest.mark.parametrize('dims', [2, 32])
@pytest.mark.parametrize('bounds', [(-1, 1), (-10, 10)])
@pytest.mark.parametrize('n', [1, 10, 100])
def test_log_likelihood(Model, dims, bounds, n):
    """Test the log-likelihood method"""
    model = create_autospec(Model)
    model.names = [f'x_{i}' for i in range(dims)]
    model.bounds = {name: bounds for name in model.names}
    x = numpy_array_to_live_points(
        np.random.uniform(low=bounds[0], high=bounds[1], size=(n, dims)),
        model.names
    )
    log_l = Model.log_likelihood(model, x)
    assert log_l.ndim == 1
    assert len(log_l) == n
    assert np.isfinite(log_l).all()
