# -*- coding: utf-8 -*-
"""Example of using nessai_models with nessai.

This example uses a two-dimensional Gaussian Mixture model.
"""
from nessai.flowsampler import FlowSampler
from nessai.utils import setup_logger
from nessai_models import SlabSpike
import numpy as np

# Set up the logger as normal for nessai
setup_logger()

model = SlabSpike(weights=[0.01, 0.99])

# We then create an instance of the sampler as usual
fs = FlowSampler(
    model, output="slab_spike", resume=False, importance_nested_sampler=True
)
analytic_evidence = -np.log(
    np.product(list(map(lambda x: (x[1] - x[0]), model.bounds.values())))
)

# And run the sampler
fs.run()
print(f"analytic logZ = {analytic_evidence}, estimated logZ={fs.logZ}")
print(f"Difference = {analytic_evidence - fs.logZ}")
