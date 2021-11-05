# -*- coding: utf-8 -*-
"""
Tests specific to the n-dimensional Gaussian.
"""
import numpy as np
from nessai.livepoint import numpy_array_to_live_points, live_points_to_array
import pytest
from scipy.stats import multivariate_normal
from unittest.mock import create_autospec, patch

from nessai_models import Gaussian
from nessai_models.gaussian import compute_gaussian_ln_evidence


@pytest.fixture
def model():
    return create_autospec(Gaussian)


@pytest.fixture
def points(request):
    n = request.param
    x = 20 * np.random.rand(10, n) - 10
    return numpy_array_to_live_points(x, [f'x_{i}' for i in range(n)])


def test_init(model):
    """Test the init method"""
    with patch('nessai_models.base.NDimensionalModel.__init__') as m:
        Gaussian.__init__(model, 2, [-5, 5])
    m.assert_called_once_with(2, [-5, 5])


@pytest.mark.parametrize('points', [2, 4, 8], indirect=True)
def test_log_likelihood(model, points):
    """Test the log-likelihood"""
    model.names = list(points.dtype.names[:-3])
    log_l = Gaussian.log_likelihood(model, points)
    x = live_points_to_array(points, names=model.names)
    target = multivariate_normal(mean=len(model.names) * [0]).logpdf(x)
    np.testing.assert_array_almost_equal(log_l, target)


@pytest.mark.parametrize(
    "dims, bounds, expected",
    [
        [2, [-10, 10], - 2 * np.log(20)],
        [None, [[-10, 10], [-10, 10]], -2 * np.log(20)],
        [4, [-5, 5], -4 * np.log(10)],
    ]
)
def test_gaussian_ln_evidence(dims, bounds, expected):
    """Assert the correct log evidence is returned."""
    out = compute_gaussian_ln_evidence(bounds, dims)
    assert expected == out
