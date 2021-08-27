# -*- coding: utf-8 -*-
"""Basic tests for all models."""
import pytest

from nessaimodels import (
    Gaussian,
    Rosenbrock
)

models = [
    Gaussian,
    Rosenbrock
]


@pytest.fixture(params=models)
def Model(request):
    return request.param


@pytest.mark.parametrize('n', [1, 10])
def test_model_likelihood_and_prior(Model, n):
    """Check the log-prior and log-likelihood integration.

    Asserts values are returned and the correct size.
    """
    model = Model()
    x = model.new_point(n)
    log_p = model.log_prior(x)
    log_l = model.log_likelihood(x)
    assert log_p.size == n
    assert log_l.size == n
