# -*- coding: utf-8 -*-
"""
Base models that remove the need to repeat code between models.
"""
from typing import Sequence, Union

from nessai.model import Model
import numpy as np


class BaseModel(Model):
    """Model that includes an evidence attribute.

    Attributes
    ----------
    ln_evidence : float
        Natural log-evidence. Not set by default.
    """
    ln_evidence: float = None


class NDimensionalModel(BaseModel):
    """Model with basic init for n-dimensional likelihoods.

    Parameters
    ----------
    dims : int
        Number of dimensions.
    bounds : Union[Sequence[float], numpy.ndarray]
        Prior bounds.
    """
    def __init__(
        self,
        dims: int,
        bounds: Union[Sequence[float], np.ndarray]
    ) -> None:
        self.names = [f'x_{i}' for i in range(dims)]
        if isinstance(bounds, (Sequence, np.ndarray)):
            if len(bounds) == 2:
                bounds = np.asarray(bounds, dtype=float)
                self.bounds = {n: bounds for n in self.names}
            else:
                raise ValueError('bounds must have length 2.')
        else:
            raise TypeError('Invalid type for `bounds` argument.')


class UniformPriorMixin:
    """Mixin class that defines a uniform prior."""

    def log_prior(self, x: np.ndarray) -> np.ndarray:
        """Log probability for a uniform prior.

        Also checks if samples are within the prior bounds.

        Parameters
        ----------
        x : numpy.ndarray
            Array of samples.

        Returns
        -------
        numpy.ndarray
            Array of log-probabilities.
        """
        with np.errstate(divide='ignore'):
            log_p = np.log(self.in_bounds(x), dtype=float)
        log_p -= np.sum(np.log(self.upper_bounds - self.lower_bounds))
        return log_p

    def to_unit_hypercube(self, x: np.ndarray) -> np.ndarray:
        """Convert the samples to the unit-hypercube.

        Parameters
        ----------
        x : numpy.ndarray
            Array of samples.

        Returns
        -------
        numpy.ndarray
            Array of rescaled samples.
        """
        x_out = x.copy()
        for n in self.names:
            x_out[n] = (
                (x[n] - self.bounds[n][0])
                / (self.bounds[n][1] - self.bounds[n][0])
            )
        return x_out

    def from_unit_hypercube(self, x: np.ndarray) -> np.ndarray:
        """Convert samples from the unit-hypercube to the prior space.

        Parameters
        ----------
        x : numpy.ndarray
            Array of samples in the unit-hypercube.

        Returns
        -------
        numpy.ndarray
            Array of sample in the prior space.
        """
        x_out = x.copy()
        for n in self.names:
            x_out[n] = (
                (self.bounds[n][1] - self.bounds[n][0])
                * x[n] + self.bounds[n][0]
            )
        return x_out
