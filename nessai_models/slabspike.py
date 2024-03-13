from .gaussianmixture import GaussianMixture
import numpy as np


class SlabSpike(GaussianMixture):
    def __init__(self, dims=3, spike_scale=1e-3, **kwargs):
        if "config" not in kwargs.keys():
            mu = np.zeros(dims)
            cov_slab = np.diag(np.ones(dims))
            cov_spike = np.diag(np.ones(dims) * spike_scale)
            kwargs["config"] = [
                dict(mean=mu, cov=cov_slab),
                dict(mean=mu, cov=cov_spike),
            ]
        return super().__init__(dims=dims, **kwargs)
