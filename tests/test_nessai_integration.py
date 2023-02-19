# -*- coding: utf-8 -*-
"""Test integration with nessai."""
import os
from nessai.flowsampler import FlowSampler
import pytest


@pytest.mark.slow_integration_test
def test_sampling(tmp_path, ModelClass):
    """Run a basic sampling run.

    Only runs 500 iterations since this should be enough to test the core
    functionality.
    """
    output = tmp_path / "test_sampling"
    output.mkdir()
    fs = FlowSampler(
        ModelClass(),
        nlive=100,
        poolsize=100,
        output=output,
        resume=False,
        max_iteration=500,
        stopping=1.0,
        plot=False,
    )
    fs.run(plot=False)
    # Make sure a result is produced
    os.path.exists(os.path.join(output, "result.json"))
