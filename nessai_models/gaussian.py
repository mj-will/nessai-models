# -*- coding: utf-8 -*-
"""
N-dimensional Gaussian likelihood
"""
from typing import Optional, Sequence, Union
import warnings

import numpy as np
from scipy.stats import multivariate_normal

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
                "dims must be specified if bounds is 1-dimensional"
            )
        ln_z = -dims * np.log(np.ptp(bounds))
    else:
        if dims and np.shape(bounds)[0] != dims:
            raise ValueError(
                "When providing 2-d bounds and dims, dims must match the "
                "first dimension of bounds."
            )
        ln_z = -np.log(np.ptp(bounds, axis=-1)).sum()
    return ln_z


class Gaussian(UniformPriorMixin, NDimensionalModel):
    """A simple n-dimensional Guassian with uniform priors.

    Defaults to a unit Gaussian two dimensions and priors defined on
    [-10, 10]^n.

    Parameters
    ----------
    dims : int
        Number of dimensions.
    bounds : Union[Sequence[float], numpy.ndarray]
        Prior bounds.
    mean : Optional[Sequence[float], numpy.ndarray]
        Mean of the Gaussian.
    cov: Optional[Sequence[float], numpy.ndarray]
        Covariance matrix of the Gaussian.
    normalise : bool
        If true, the log-likelihood will be renormalised such that the log-
        evidence is zero. Only applies when :code:`mean` and :code:`cov` are
        not specified.
    """

    def __init__(
        self,
        dims: int = 2,
        bounds: Union[Sequence[float], np.ndarray] = [-10.0, 10.0],
        mean: Union[Sequence[float], np.ndarray] = None,
        cov: Union[Sequence[float], np.ndarray] = None,
        normalise: bool = False,
    ) -> None:
        super().__init__(dims, bounds)

        self._norm_const = 0.0
        if mean is None:
            self.mean = np.zeros(self.dims)
        elif isinstance(mean, (float, int)):
            self.mean = float(mean) * np.ones(self.dims)
        else:
            self.mean = mean

        if cov is None:
            self.cov = np.eye(self.dims)
        else:
            self.cov = cov

        self.dist = multivariate_normal(mean=self.mean, cov=self.cov)
        self.normalise = normalise

        if cov is None and mean is None:
            self.ln_evidence = compute_gaussian_ln_evidence(
                bounds, dims=self.dims
            )
            if self.normalise:
                self._norm_const = self.ln_evidence
        else:
            self.ln_evidence = None
            if self.normalise:
                warnings.warn("Cannot normalise non-unit Gaussian")
                self.normalise = False

    def log_likelihood(self, x: np.ndarray) -> np.ndarray:
        """Gaussian log-likelihood."""
        # Use a view rather than making a new copy of y
        return self.dist.logpdf(self.unstructured_view(x)) - self._norm_const
