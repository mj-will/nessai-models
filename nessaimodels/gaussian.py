# -*- coding: utf-8 -*-
"""
N-dimensional Gaussian likelihood
"""
from typing import Optional, Sequence, Union

import numpy as np
from scipy.stats import norm

from .base import NDimensionalModel, UniformPriorMixin


def compute_gaussian_ln_evidence(
    bounds: Union[list, tuple, np.ndarray],
    dims: Optional[int] = None,
) -> float:
    """Compute the ln-evidence for a unit Gaussian likelihood:

    Parameters
    ----------
    bounds : Union[list, tuple numpy.ndarray]
        Prior bounds. Either 1-d or 2-d. When 1-d input is given, `dims` must
        be specified and the prior bounds are assumed to be the same in each
        dimension.
    dims : Optional[int]
        Number of dimensions. If a 2-d bounds array is given, then the value
        of `dims` is checked against the shape of `bounds`.
    """
    if np.ndim(bounds) == 1:
        if dims is None:
            raise ValueError(
                'dims must be specified if bounds is 1-dimensional'
            )
        ln_z = -dims * np.log(np.ptp(bounds))
    else:
        if dims and np.shape(bounds)[0] != dims:
            raise ValueError(
                'When providing 2-d bounds and dims, dims must match the '
                'first dimension of bounds.'
            )
        ln_z = -np.log(np.ptp(bounds, axis=-1)).sum()
    return ln_z


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
        self.ln_evidence = compute_gaussian_ln_evidence(bounds, dims=self.dims)

    def log_likelihood(self, x: np.ndarray) -> np.ndarray:
        """Gaussian log-likelihood."""
        log_l = np.zeros(x.size)
        for n in self.names:
            log_l += norm.logpdf(x[n])
        return log_l
