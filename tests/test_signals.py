"""Tests for signal-based models"""
import numpy as np
import pytest

from nessai_models.signals import (
    LinearSignal,
    SinusoidalSignal,
)

all_models = [
    LinearSignal,
    SinusoidalSignal,
]


@pytest.fixture(params=all_models)
def SignalModelClass(request):
    """Signal model classes fixture."""
    return request.param


@pytest.mark.parametrize("n_points", [10, 100])
@pytest.mark.parametrize("sigma", [1, 2])
def test_max_likelihood(SignalModelClass, n_points, sigma):
    """Assert the maximum likelihood is the correct value.

    Sets the data to have zero noise.
    """
    model = SignalModelClass(n_points=n_points, sigma=sigma)
    model.data = model.signal_model(**model.truth)
    expected = n_points * -np.log(2 * np.pi * sigma**2)
    actual = model.log_likelihood(model.truth)
    np.testing.assert_almost_equal(actual, expected, decimal=12)
