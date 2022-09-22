"""
N-dimensional Gaussian likelihood
"""
from typing import Sequence, Union

import numpy as np
from scipy.stats import halfnorm

from .base import NDimensionalModel, UniformPriorMixin
from .gaussian import compute_gaussian_ln_evidence


class HalfGaussian(UniformPriorMixin, NDimensionalModel):
    """A multi-dimensional Half Gaussian likelihood.

    The lower bound must be set to zero in all dimensions.

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
        bounds: Union[Sequence[int], np.ndarray] = [0.0, 10.0],
    ) -> None:
        super().__init__(dims, bounds)
        if not all(self.lower_bounds == 0.0):
            raise ValueError("Lower bounds must all be zero!")
        self.ln_evidence = compute_gaussian_ln_evidence(bounds, dims=self.dims)

    def log_likelihood(self, x: np.ndarray) -> np.ndarray:
        """Gaussian log-likelihood."""
        log_l = np.zeros(x.size)
        for n in self.names:
            log_l += halfnorm.logpdf(x[n])
        return log_l
