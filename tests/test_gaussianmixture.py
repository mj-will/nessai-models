# -*- coding: utf-8 -*-
"""
Tests specific to the n-dimensional Gaussian Mixture models.
"""
from nessai.livepoint import numpy_array_to_live_points
import numpy as np
from scipy.stats import multivariate_normal
import pytest

from nessai_models.gaussianmixture import GaussianMixture


@pytest.mark.integration_test
def test_weighted_gm():
    """Integration test to verify the log-likelihood."""
    dims = 2
    n_points = 10
    weights = [0.1, 0.9]
    n_gaussians = 2
    config = [
        dict(mean=dims * [-1], cov=2 * np.eye(dims)),
        dict(mean=dims * [1], cov=2 * np.eye(dims)),
    ]
    model = GaussianMixture(
        dims=dims,
        weights=weights,
        config=config,
        n_gaussians=n_gaussians,
    )

    dist0 = multivariate_normal(
        mean=config[0]["mean"],
        cov=config[0]["cov"],
    )
    dist1 = multivariate_normal(
        mean=config[1]["mean"],
        cov=config[1]["cov"],
    )
    x = np.random.randn(n_points, dims)
    x_live = numpy_array_to_live_points(x, model.names)
    expected = np.log(weights[0] * dist0.pdf(x) + weights[1] * dist1.pdf(x))

    out = model.log_likelihood(x_live)
    assert out.shape == (n_points,)

    np.testing.assert_array_almost_equal_nulp(out, expected)
