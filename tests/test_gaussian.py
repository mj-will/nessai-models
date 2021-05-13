# -*- coding: utf-8 -*-
"""
Test the n-dimensional Gaussian
"""
import numpy as np
from nessai.livepoint import numpy_array_to_live_points, live_points_to_array
import pytest
from scipy.stats import multivariate_normal
from unittest.mock import create_autospec

from nessaimodels import Gaussian


@pytest.fixture
def model():
    return create_autospec(Gaussian)


@pytest.fixture
def points(request):
    n = request.param
    x = 20 * np.random.rand(10, n) - 10
    return numpy_array_to_live_points(x, [f'x_{i}' for i in range(n)])


@pytest.mark.parametrize('n', [2, 4, 8])
def test_init(n):
    """Test the init method"""
    g = Gaussian(n)
    assert len(g.names) == n
    assert len(g.bounds.keys()) == n
    assert g.bounds == {p: [-10, 10] for p in g.names}


@pytest.mark.parametrize('points', [2, 4, 8], indirect=True)
def test_log_prior(model, points):
    """Test the log_prior"""
    model.names = points.dtype.names[:-3]
    model.bounds = {n: [-10.0, 10.0] for n in model.names}
    log_p = Gaussian.log_prior(model, points)
    target = -len(model.names) * np.log(20.0)
    np.testing.assert_array_equal(log_p, target)


@pytest.mark.parametrize('points', [2, 4, 8], indirect=True)
def test_log_likelihood(model, points):
    """Test the log_prior"""
    model.names = list(points.dtype.names[:-3])
    model.bounds = {n: [-10.0, 10.0] for n in model.names}
    log_l = Gaussian.log_likelihood(model, points)
    x = live_points_to_array(points, names=model.names)
    target = multivariate_normal(mean=len(model.names) * [0]).logpdf(x)
    np.testing.assert_array_almost_equal(log_l, target)
