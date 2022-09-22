# -*- coding: utf-8 -*-
"""
N-dimensional pyramid likelihood
"""
from typing import Sequence, Union

import numpy as np
from .base import NDimensionalModel, UniformPriorMixin


class Pyramid(UniformPriorMixin, NDimensionalModel):
    """N-dimensional pyramid likelihood with uniform priors.

    Parameters
    ----------
    dims :
        Number of dimensions.
    bounds :
        Prior bounds.
    """

    def __init__(
        self,
        dims: int = 2,
        bounds: Union[Sequence[int], np.ndarray] = [-10, 10],
    ) -> None:
        super().__init__(dims, bounds)

    def log_likelihood(self, x: np.ndarray) -> np.ndarray:
        """Compute the log-likelihood.

        Parameters
        ----------
        x :
            Point or array of points as a structured array with fields
            that match the names of the model.

        Returns
        -------
        numpy.ndarray
            One-dimensional array of log-likelihoods
        """
        return -np.sum(np.abs(self.unstructured_view(x)), axis=-1)
