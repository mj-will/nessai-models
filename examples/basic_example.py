# -*- coding: utf-8 -*-
"""Example of using nessai_models with nessai.

This example uses a two-dimensional Gaussian Mixture model.
"""
from nessai.flowsampler import FlowSampler
from nessai.utils import setup_logger
from nessai_models import GaussianMixture

# Set up the logger as normal for nessai
setup_logger()

# The GaussianMixture model from nessai_models already has a parameter names,
# prior bounds, the log-prior and log-likelihood defined. So all we need to
# do is create an instance of class and, optionally, specify the number of
# dimensions.
# This particular model as has extra parameters that default values, for
# example the number of Gaussian in the mixture. By default, there weights and
# parameters (mean and covariance) will be randomised.
model = GaussianMixture(dims=2)

# We then create an instance of the sampler as usual
fs = FlowSampler(model, output="outdir", resume=False)
# And run the sampler
fs.run()
