# -*- coding: utf-8 -*-
"""
N-dimensional Gaussian likelihood
"""
from typing import Sequence, Union

import numpy as np
from scipy.stats import norm

from .base import NDimensionalModel, UniformPriorMixin


class Gaussian(UniformPriorMixin, NDimensionalModel):
    """A simple n-dimensional unit Guassian.

    Defaults to two dimensions and priors defined on [-10, 10]^n.

    Parameters
    ----------
    dims : int
        Number of dimensions.
    bounds : Union[Sequence[float], numpy.ndarray]
        Prior bounds.
    """
    def __init__(
        self,
        dims: int = 2,
        bounds: Union[Sequence[int], np.ndarray] = [-10.0, 10.0]
    ) -> None:
        super().__init__(dims, bounds)

    def log_likelihood(self, x: np.ndarray) -> np.ndarray:
        """Gaussian log-likelihood."""
        log_l = np.zeros(x.size)
        for n in self.names:
            log_l += norm.logpdf(x[n])
        return log_l
