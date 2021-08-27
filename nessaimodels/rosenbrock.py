
# -*- coding: utf-8 -*-
"""
N-dimensional Rosenbrock likelihood
"""
from typing import Sequence, Union

from nessai.livepoint import live_points_to_array
import numpy as np

from .base import NDimensionalModel, UniformPriorMixin


class Rosenbrock(UniformPriorMixin, NDimensionalModel):
    """An n-dimensional Rosenbrock likelihood.

    Defaults to two dimensions and priors defined on [-5, 5]^n.

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
        bounds: Union[Sequence[int], np.ndarray] = [-5.0, 5.0]
    ) -> None:
        super().__init__(dims, bounds)

    def log_likelihood(self, x: np.ndarray) -> np.ndarray:
        """Rosenbrock Log-likelihood."""
        x = live_points_to_array(x, self.names)
        x = np.atleast_2d(x)
        return -np.sum(
            100. * (x[:, 1:] - x[:, :-1] ** 2.0) ** 2.0
            + (1.0 - x[:, :-1]) ** 2.0,
            axis=1
        )
