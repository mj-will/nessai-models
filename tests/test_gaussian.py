# -*- coding: utf-8 -*-
"""
Tests specific to the n-dimensional Gaussian.
"""
import numpy as np
from nessai.livepoint import numpy_array_to_live_points, live_points_to_array
import pytest
from scipy.stats import multivariate_normal
from unittest.mock import MagicMock, create_autospec, patch

from nessai_models import Gaussian
from nessai_models.gaussian import compute_gaussian_ln_evidence


@pytest.fixture
def model():
    return create_autospec(Gaussian)


def test_init(model):
    """Test the init method"""
    model.dims = 2
    with patch('nessai_models.base.NDimensionalModel.__init__') as m, \
         patch(
            'nessai_models.gaussian.compute_gaussian_ln_evidence',
            return_value=1.0
        ) as m1:
        Gaussian.__init__(model, 2, [-5, 5], normalise=True)
    m.assert_called_once_with(2, [-5, 5])
    m1.assert_called_once_with([-5, 5], dims=2)
    assert model.ln_evidence == 1.0
    assert model._norm_const == 1.0


def test_init_normalise_false(model):
    """Assert the normalisation constant is zero when normalise is False"""
    model.dims = 2
    Gaussian.__init__(model, 2, [-5, 5], normalise=False, cov=None, mean=None)
    assert model._norm_const == 0.0


def test_init_cannot_normalise(model):
    """Assert the normalisation constant is zero if normalisation is not
    possible.
    """
    mean = 1
    cov = np.eye(2)
    model.dims = 2
    with pytest.warns(Warning) as warninfo:
        Gaussian.__init__(
            model, 2, [-5, 5], mean=mean, cov=cov, normalise=True
        )
    assert model._norm_const == 0.0
    assert "Cannot normalise" in str(warninfo[0].message)


@pytest.mark.parametrize(
    "mean, expected",
    [
        (None, np.zeros(2)),
        (2.0, np.array([2, 2])),
        (np.array([2, 2]), np.array([2, 2])),
    ]
)
def test_init_mean(model, mean, expected):
    """Assert the mean is set correctly"""
    model.dims = len(expected)
    Gaussian.__init__(model, len(expected), [-5, 5], mean=mean)
    np.testing.assert_equal(model.mean, expected)


@pytest.mark.parametrize(
    "cov, expected",
    [
        (None, np.eye(2)),
        (np.eye(2), np.eye(2)),
    ]
)
def test_init_cov(model, cov, expected):
    """Assert the covariance is set correctly"""
    model.dims = len(expected)
    Gaussian.__init__(model, len(expected), [-5, 5], cov=cov)
    np.testing.assert_equal(model.cov, expected)


def test_log_likelihood(model):
    """Test the log-likelihood"""
    x = np.random.randn(4, 2)
    x_view = np.random.randn(4, 2)
    log_l = np.array([1, 2, 3, 4])
    model.unstructured_view = MagicMock(return_value=x_view)
    model.dist = MagicMock()
    model.dist.logpdf = MagicMock(return_value=log_l)
    model._norm_const = 1

    out = Gaussian.log_likelihood(model, x)
    np.testing.assert_equal(out, np.array([0, 1, 2, 3]))

    model.unstructured_view.assert_called_once_with(x)
    model.dist.logpdf.assert_called_once_with(x_view)


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


def test_gaussian_ln_evidence_dims_error():
    """Assert an error is raised if the bounds are 1-d and dims is None."""
    with pytest.raises(ValueError) as excinfo:
        compute_gaussian_ln_evidence([-5, 5])
    assert "dims must be specified" in str(excinfo.value)


def test_gaussian_ln_evidence_dims_wrong():
    """Assert an error is raised if the dims and bounds do not agree."""
    with pytest.raises(ValueError) as excinfo:
        compute_gaussian_ln_evidence(np.array([[-5, 5], [-5, 5]]), 3)
    assert "dims must match the first dimension" in str(excinfo.value)
