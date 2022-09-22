# -*- coding: utf-8 -*-
"""
N-dimensional egg box likelihood as defined in Feroz et al. 2008.
"""
from typing import Sequence, Union

import numpy as np
from .base import NDimensionalModel, UniformPriorMixin


class EggBox(UniformPriorMixin, NDimensionalModel):
    """N-dimensional egg box based of the version in \
        `Feroz et al. 2008 <https://arxiv.org/abs/0809.3437>`_

    This model extends the two-dimensional version of the Egg Box defined in
    Feroz at al. 2008 to n-dimensions as:

    .. math::

        \\log \\mathcal{L} =
        \\left[ 2 + \\prod_{i=1}^{n} \\cos \\frac{ \\theta_i}{2} \\right] ^ {5}

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
        bounds: Union[Sequence[int], np.ndarray] = [0, 10.0 * np.pi],
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
        log_l = np.ones(x.size)
        for n in self.names:
            log_l += np.cos(x[n] / 2.0)
        return (log_l + 2.0) ** 5.0
