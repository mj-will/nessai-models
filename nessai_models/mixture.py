# -*- coding: utf-8 -*-
"""
Likelihoods that are mixtures of distributions.
"""
from functools import partial
from typing import Callable, Optional

import numpy as np
from scipy import stats

from .base import BaseModel, UniformPriorMixin


class MixtureOfDistributions(UniformPriorMixin, BaseModel):
    """Mixture of distributions.

    Available distributions: Gaussian, Uniform, Gamma, HalfNorm

    Parameters
    ----------
    distributions
        Dictionary with the names of the distributions as keys and the number
        of each distribution to add as the values.
    bounds
        Dictionary of priors bounds
    distributions_kwargs
        Dictionary of dictionaries where the key is the name of distribution
        and the values are a dictionary of keyword arguments.
    map_fn
        Map function use when computing the likelihood.
    """

    def __init__(
        self,
        distributions: Optional[dict] = None,
        bounds: Optional[dict] = None,
        distributions_kwargs: Optional[dict] = None,
        map_fn: Optional[Callable] = None,
    ) -> None:
        self.bounds_mapping = dict(
            gaussian=[-10.0, 10.0],
            uniform=[-5.0, 5.0],
            gamma=[0.0, 10.0],
            halfnorm=[0.0, 10.0],
        )

        if distributions is None:
            distributions = {"gaussian": 2, "uniform": 2, "gamma": 2}

        if distributions_kwargs:
            if set(distributions_kwargs).difference(self.bounds_mapping):
                raise ValueError(
                    "distribution_kwargs contains unknown distributions"
                )
        else:
            distributions_kwargs = {}

        self.base_mapping = dict(
            gaussian=stats.norm(
                **distributions_kwargs.get("gaussian", {})
            ).logpdf,
            uniform=stats.uniform(
                self.bounds_mapping["uniform"][0],
                np.ptp(self.bounds_mapping["uniform"]),
                **distributions_kwargs.get("uniform", {}),
            ).logpdf,
            gamma=stats.gamma(
                **distributions_kwargs.get("gamma", {"a": 1.99})
            ).logpdf,
            halfnorm=stats.halfnorm(
                **distributions_kwargs.get("halfnorm", {})
            ).logpdf,
        )

        self.mapping = dict()
        if bounds is None:
            bounds = dict()

        for dist, n in distributions.items():
            for i in range(n):
                name = dist + f"_{i}"
                self.mapping[name] = self.base_mapping[dist.lower()]
                if name not in bounds:
                    bounds[name] = self.bounds_mapping[dist]

        self.names = list(self.mapping.keys())
        self.bounds = bounds

        if map_fn is None:
            self.map_fn = map
        else:
            self.map_fn = map_fn

    @staticmethod
    def _log_likelihood_name(
        mapping: dict, x: np.ndarray, name: str
    ) -> np.ndarray:
        return mapping[name](x[name])

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
        return np.vstack(
            list(
                self.map_fn(
                    partial(self._log_likelihood_name, self.mapping, x),
                    self.names,
                )
            )
        ).sum(axis=0)
