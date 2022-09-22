# -*- coding: utf-8 -*-
"""
N-dimensional Rosenbrock likelihood
"""
from typing import Sequence, Union

import numpy as np

from .base import NDimensionalModel, UniformPriorMixin


def uncoupled_rosenbrock(x: np.ndarray) -> np.ndarray:
    r"""Uncoupled Rosenbrock function in N dimensions.

    This is the simpler version which is the sum of N/2 uncouple 2D
    Rosenbrocks given by

    .. math::
        \sum_{i=1}^{N/2} [100(x_{2i-1}^{2} - x_{2i})^2 + (x_{2i-1} - 1)^2].
    """
    return np.sum(
        100.0 * (x[..., ::2] ** 2.0 - x[..., 1::2]) ** 2.0
        + (x[..., ::2] - 1.0) ** 2.0,
        axis=-1,
    )


def rosenbrock(x: np.ndarray) -> np.ndarray:
    r"""Rosenbrock function in N dimensions.

    This is the more involved variant given by

    .. math::
        \sum_{i=1}^{N-1} [100(x_{i+1} - x_{i}^{2})^2 + (1 - x_{i})^2].
    """
    return np.sum(
        100.0 * (x[..., 1:] - x[..., :-1] ** 2.0) ** 2.0
        + (1.0 - x[..., :-1]) ** 2.0,
        axis=-1,
    )


class Rosenbrock(UniformPriorMixin, NDimensionalModel):
    """An n-dimensional Rosenbrock likelihood.

    Defaults to two dimensions and priors defined on [-5, 5]^n.

    Parameters
    ----------
    dims : int
        Number of dimensions.
    bounds : Union[Sequence[float], numpy.ndarray]
        Prior bounds.
    uncouple : bool
        Enable the uncoupled (simpler) version of the Rosenbrock likelihood.
    """

    def __init__(
        self,
        dims: int = 2,
        bounds: Union[Sequence[int], np.ndarray] = [-5.0, 5.0],
        uncoupled: float = False,
    ) -> None:
        super().__init__(dims, bounds)

        self.uncoupled = uncoupled
        if self.uncoupled:
            self._fn = uncoupled_rosenbrock
        else:
            self._fn = rosenbrock

    def log_likelihood(self, x: np.ndarray) -> np.ndarray:
        """Rosenbrock Log-likelihood."""
        return -self._fn(self.unstructured_view(x))
