# -*- coding: utf-8 -*-
"""
Specific tests for the Pyramid model.
"""
from nessai.livepoint import parameters_to_live_point
import pytest

from nessai_models.pyramid import Pyramid


@pytest.mark.parametrize("dims", [2, 4, 8, 16])
@pytest.mark.integration_test
def test_max_log_likelihood(dims):
    """Assert the maximum log-likelihood is zero at [0, 0]"""
    model = Pyramid(dims=dims)
    x = parameters_to_live_point(dims * [0.0], [f'x{i}' for i in range(dims)])
    assert model.log_likelihood(x) == 0.0
