"""Tests the for the mixture module."""
import multiprocessing.dummy as mp
from nessai_models.mixture import MixtureOfDistributions
import numpy as np

import pytest


@pytest.mark.integration_test
def test_multiprocessing_pool():
    """Assert the map function fom Pool can be used.""" 
    pool = mp.Pool(2)
    model = MixtureOfDistributions(map_fn=pool.map)
    x = model.new_point(100)
    model.log_likelihood(x)
    pool.close()


@pytest.mark.parametrize(
    "distributions",
    [{"gaussian": 2}, {"gamma": 2}, {"uniform": 2}, {"halfnorm": 2}]
)
def test_distributions(distributions):
    """Test the init and likelihood with different distributions"""
    model = MixtureOfDistributions(distributions=distributions)
    x = model.new_point()
    out = model.log_likelihood(x)
    assert np.isfinite(out).all()

