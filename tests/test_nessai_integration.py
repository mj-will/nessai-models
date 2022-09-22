# -*- coding: utf-8 -*-
"""Test integration with nessai."""
import os

from nessai.flowsampler import FlowSampler
from nessai_models import Gaussian
import pytest


@pytest.mark.slow_integration_test
def test_sampling(tmp_path):
    """Run a basic sampling run"""
    output = tmp_path / "test_sampling"
    output.mkdir()
    fs = FlowSampler(
        Gaussian(2),
        nlive=100,
        output=output,
        resume=False,
        checkpointing=False,
        tolerance=1.0,
        plot=False,
    )
    fs.run(plot=False)
    # Make sure a result is produced
    os.path.exists(os.path.join(output, "result.json"))
