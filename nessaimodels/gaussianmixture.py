# -*- coding: utf-8 -*-
"""
Gaussian mixture models.
"""
import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import logsumexp

from nessai.livepoint import live_points_to_array
from nessai.model import Model


class GaussianMixture(Model):
    """A Gaussian mixture model with n Gaussians defined on [-10, 10]^n.

    Parameters
    ----------
    dims : int, optional
        Number of dimensions
    n : int, optional
        Number of Gaussians.
    config : list of dict, optional
        List of configurations for each Gaussian. Each dictionary should have
        a mean and cov key.
    random_state : :obj:`nessai.random.RandomState`, optional
        Random state to use for generation configuration if `config` is not
        specified. If not specified `seed` is used instead.
    seed : int, optional
        Random seed for seeding random number generation.
    """
    def __init__(self, dims=4, n_gaussians=2, config=None, random_state=None,
                 seed=1234, bounds=[-10, 10]):
        self.names = [f'x_{i}' for i in range(dims)]
        self.bounds = {n: bounds for n in self.names}

        if random_state is None:
            random_state = np.random.RandomState(seed=seed)

        self.n_gaussians = n_gaussians
        self.gaussians = n_gaussians * [None]
        if config is None:
            config = n_gaussians * [None]
        elif len(config) != n_gaussians:
            raise RuntimeError('Config does not match number of Gaussians')

        for n in range(n_gaussians):
            if config[n] is None:
                config[n] = dict(
                    mean=random_state.uniform(bounds[0], bounds[1], dims),
                    cov=3 * random_state.rand() * np.eye(dims)
                )
            self.gaussians[n] = multivariate_normal(**config[n])

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
        _x = live_points_to_array(x, self.names)
        log_l = logsumexp([g.logpdf(_x) for g in self.gaussians], axis=0)
        return log_l


class GaussianMixtureWithData(Model):
    """
    A Gaussian mixture model with two peaks that uses samples and fits the
    the means standard deviations and weight.

    Based on the example from cpnest: \
        https://github.com/johnveitch/cpnest/blob/master/examples/gaussianmixture.py

    Parameters
    ----------
    n : int, optional
        Number of data points to use.
    """
    def __init__(self, n=10000):
        self.names = ['mu1', 'sigma1', 'mu2', 'sigma2', 'weight']
        self.bounds = {
            'mu1': [-3, 3],
            'sigma1': [0.01, 1.0],
            'mu2': [-3, 3],
            'sigma2': [0.01, 1.0],
            'weight': [0.01, 0.99]
        }

        self.truth = {
            'mu1': 0.5,
            'sigma1': 0.5,
            'mu2': -1.5,
            'sigma2': 0.03,
            'weight': 0.1
        }
        n1 = int(self.truth['weight'] * n)
        n2 = n - n1

        self.data = np.concatenate([
            np.random.normal(self.truth['mu1'], self.truth['sigma1'], size=n1),
            np.random.normal(self.truth['mu2'], self.truth['sigma2'], size=n2)
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
        for n in ['sigma1', 'sigma2']:
            log_p -= np.log(x[n])
        return log_p

    def log_likelihood(self, x):
        """
        Returns log likelihood of given live point.
        """
        w = x['weight']
        log_l1 = (np.log(w) - np.log(x['sigma1']) -
                  0.5 * ((self.data - x['mu1']) / x['sigma1']) ** 2)
        log_l2 = (np.log(1.0 - w) - np.log(x['sigma2']) -
                  0.5 * ((self.data - x['mu2']) / x['sigma2']) ** 2)
        log_l = np.logaddexp(log_l1, log_l2).sum()
        return log_l
