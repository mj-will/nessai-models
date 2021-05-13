# -*- coding: utf-8 -*-
"""
N-dimensional Gaussian likelihood
"""
import numpy as np
from scipy.stats import norm

from nessai.model import Model


class Gaussian(Model):
    """
    A simple n-dimensional Guassian likelihood
    """
    def __init__(self, n=2, bounds=[-10.0, 10.0]):
        self.names = [f'x_{i}' for i in range(n)]
        self.bounds = {p: bounds for p in self.names}

    def log_prior(self, x):
        """
        Returns log of prior given a live point assuming uniforn
        priors on each parameter.
        """
        log_p = np.zeros(x.size)
        for n in self.names:
            log_p += np.log((x[n] >= self.bounds[n][0])
                            & (x[n] <= self.bounds[n][1]))
            log_p -= np.log(self.bounds[n][1] - self.bounds[n][0])
        return log_p

    def log_likelihood(self, x):
        """
        Returns log likelihood of given live point assuming a Gaussian
        likelihood.
        """
        log_l = np.zeros(x.size)
        for n in self.names:
            log_l += norm.logpdf(x[n])
        return log_l
