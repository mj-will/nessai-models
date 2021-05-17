
# -*- coding: utf-8 -*-
"""
N-dimensional Rosenbrock likelihood
"""
import numpy as np

from nessai.livepoint import live_points_to_array
from nessai.model import Model


class Rosenbrock(Model):
    """
    An n-dimensional Rosenbrock likelihood,
    """
    def __init__(self, n=2, bounds=[-5.0, 5.0]):
        self.names = [f'x_{i}' for i in range(n)]
        self.bounds = {p: bounds for p in self.names}

    def log_prior(self, x):
        """Uniform prior"""
        log_p = np.zeros(x.size)
        for n in self.names:
            log_p += np.log((x[n] >= self.bounds[n][0])
                            & (x[n] <= self.bounds[n][1]))
            log_p -= np.log(self.bounds[n][1] - self.bounds[n][0])
        return log_p

    def log_likelihood(self, x):
        """Rosenbrock Log-likelihood."""
        x = live_points_to_array(x, self.names)
        x = np.atleast_2d(x)
        return -np.sum(
            100. * (x[:, 1:] - x[:, :-1] ** 2.0) ** 2.0
            + (1.0 - x[:, :-1]) ** 2.0,
            axis=1
        )
