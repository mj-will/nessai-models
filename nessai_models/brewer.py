# -*- coding: utf-8 -*-
"""
Likelihood described in Brewer et al. arXiv:0912.2380
"""
from typing import Sequence, Union

import numpy as np
from scipy.stats import multivariate_normal

from .base import NDimensionalModel, UniformPriorMixin


class Brewer(UniformPriorMixin, NDimensionalModel):
    """Bimodal likelihood described in Brewer et al. arXiv:0912.2380.

    Defaults match those described in the paper, but can be changed to match
    Skilling's original "Statistics Problem".

    Parameters
    ----------
    v_mean : float
        Mean of the wider peak.
    u_mean : float
        Mean of the narrower peak.
    v_width : float
        Width of the wider peak.
    u_width : float
        Width of the narrow peak.
    weight : float
        Relative weight of the narrower peak compared to the wider peak.
    dims : int
        Number of dimensions.
    bounds : Union[Sequence[float], numpy.ndarray]
        Prior bounds.
    """

    def __init__(
        self,
        v_mean: float = 0.0,
        u_mean: float = 0.031,
        v_width: float = 0.1,
        u_width: float = 0.01,
        weight: float = 100.0,
        dims: int = 20,
        bounds: Union[Sequence[float], np.ndarray] = [-0.5, 0.5],
    ) -> None:
        super().__init__(dims=dims, bounds=bounds)

        self.weight = weight
        self.ln_weight = np.log(weight)
        self.v_dist = multivariate_normal(
            mean=v_mean * np.ones(self.dims),
            cov=(v_width**2) * np.eye(self.dims),
        )
        self.u_dist = multivariate_normal(
            mean=u_mean * np.ones(self.dims),
            cov=(u_width**2) * np.eye(self.dims),
        )

    def log_likelihood(self, x: np.ndarray) -> np.ndarray:
        """Likelihood function.

        Parameters
        ----------
        x : numpy.ndarray
            Structured array of parameters.

        Returns
        -------
        numpy.ndarray
            Array of log-likelihood values.
        """
        x = self.unstructured_view(x)
        return np.logaddexp(
            self.v_dist.logpdf(x),
            self.ln_weight + self.u_dist.logpdf(x),
        )
