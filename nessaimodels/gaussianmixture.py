# -*- coding: utf-8 -*-
"""
Gaussian mixture model
"""
import numpy as np
from scipy.stats import norm

from nessai.model import Model


class GaussianMixture(Model):
    """
    A Gaussian mixture model with two peaks.

    Based on the example from cpnest: \
        https://github.com/johnveitch/cpnest/blob/master/examples/gaussianmixture.py

    The parameters to estimate are the means, standard deviations and the
    weight.

    Parameters
    ----------
    n : int, optional
        Number of data points to use.
    """
    def __init__(self, n=10000):
        self.names = ['mu1', 'sigma1', 'mu2', 'sigma2', 'weight']
        self.bounds = {
            'mu1': [-3, 3],
            'sigma1': [0.01, 1],
            'mu2': [-3, 3],
            'sigma2': [0.01, 1.0],
            'weight': [0.0, 1.0]
        }

        self.truth = {
            'mu1': 0.5,
            'sigma1': 0.5,
            'mu2': -1.5,
            'sigma2': 0.03,
            'weight': 0.2
        }
        self.gaussian1 = norm(self.truth['mu1'], scale=self.truth['sigma1'])
        self.gaussian2 = norm(self.truth['mu2'], scale=self.truth['sigma2'])

        n1 = int(self.truth['weight'] * n)
        n2 = n - n1

        self.data = np.concatenate([
            self.gaussian1.rvs(size=n1),
            self.gaussian2.rvs(size=n2)
        ])

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
        Returns log likelihood of given live point.
        """
        w = x['weight']
        log_l1 = np.sum(np.log(w) - np.log(x['sigma1']) -
                        0.5 * ((self.data - x['mu1']) / x['sigma1']) ** 2)
        log_l2 = np.sum(np.log(1.0 - w) - np.log(x['sigma2']) -
                        0.5 * ((self.data - x['mu2']) / x['sigma2']) ** 2)
        log_l = np.logaddexp(log_l1, log_l2)
        return log_l
