"""Signal plus noise models."""

from abc import abstractmethod
from typing import Dict, List, Optional

import numpy as np

from .base import BaseModel, UniformPriorMixin


class GaussianNoisePlusSignal(UniformPriorMixin, BaseModel):
    """Gaussian noise plus signal model.

    Parameters
    ----------
    names : List[str]
        Names of the parameters
    truth : dict
        Dictionary contain the true value for the injected signal.
    sigma : float
        Standard deviation of the Gaussian noise.
    bounds : Dict
        Prior bounds for the parameters.
    n_points : int
        The number of data points to use.
    start : float
        The starting x-value.
    end : float
        The ending x-value.
    """

    def __init__(
        self,
        names: List[str],
        truth: Optional[Dict] = None,
        sigma: float = 1.0,
        bounds: Dict = None,
        n_points: int = 100,
        start: float = 0.0,
        end: float = 1.0,
    ) -> None:
        self.names = names

        if truth is None:
            truth = {k: np.random.uniform(*v) for k, v in bounds.items()}
        elif list(truth.keys()) != self.names:
            raise ValueError("Keys in truth dictionary do not match names")

        self.bounds = bounds
        self.truth = truth
        self.sigma = sigma

        self.x = np.linspace(start, end, n_points)[:, np.newaxis]
        self.data = self.signal_model(
            **self.truth
        ) + self.sigma * np.random.randn(n_points, 1)

    @abstractmethod
    def signal_model(self):
        """Must be implemented by the child class.

        Should be defined using named arguments.
        """
        raise NotImplementedError

    def log_likelihood(self, x: np.ndarray) -> np.ndarray:
        """Compute the log-likelihood"""
        fits = self.signal_model(**{n: x[n] for n in self.names})
        log_l = np.sum(
            -0.5 * (((self.data - fits) / self.sigma) ** 2)
            - np.log(2 * np.pi * self.sigma**2),
            axis=0,
        )
        return log_l


class LinearSignal(GaussianNoisePlusSignal):
    """Linear signal model in Gaussian noise.

    Parameter names are: m, c
    """

    def __init__(
        self,
        truth: Optional[Dict] = None,
        sigma: float = 1,
        bounds: Optional[Dict] = None,
        n_points: int = 100,
        start: float = 0,
        end: float = 10,
    ) -> None:
        names = ["m", "c"]
        if bounds is None:
            bounds = dict(
                m=[-1, 1],
                c=[-1, 1],
            )

        super().__init__(
            names=names,
            truth=truth,
            sigma=sigma,
            bounds=bounds,
            n_points=n_points,
            start=start,
            end=end,
        )

    def signal_model(self, *, m, c) -> np.ndarray:
        """Linear signal model."""
        return m * self.x + c


class SinusoidalSignal(GaussianNoisePlusSignal):
    """Sinusoidal signal model in Gaussian noise.

    Parameter names are: amp, phase, f, offset
    """

    def __init__(
        self,
        truth: Optional[Dict] = None,
        sigma: float = 1,
        bounds: Optional[Dict] = None,
        n_points: int = 100,
        start: float = 0,
        end: float = 10,
    ) -> None:
        names = ["amp", "phase", "f", "offset"]
        if bounds is None:
            bounds = dict(
                amp=[0, 1],
                f=[0, 5],
                phase=[0, 2 * np.pi],
                offset=[0, 5],
            )

        super().__init__(
            names=names,
            truth=truth,
            sigma=sigma,
            bounds=bounds,
            n_points=n_points,
            start=start,
            end=end,
        )

    def signal_model(self, *, amp, f, phase, offset) -> np.ndarray:
        """Sinusoidal signal model."""
        return amp * np.sin(2 * np.pi * f * self.x + phase) + offset
