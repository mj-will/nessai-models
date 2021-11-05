# -*- coding: utf-8 -*-
"""
Models for use with the nested sampler \
    `nessai <https://github.com/mj-will/nessai-models>`_.
"""
try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:   # for Python < 3.8
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    pass

from .gaussian import Gaussian
from .gaussianmixture import GaussianMixture, GaussianMixtureWithData
from .halfgaussian import HalfGaussian
from .rosenbrock import Rosenbrock

__all__ = [
    'Gaussian',
    'GaussianMixture',
    'GaussianMixtureWithData',
    'HalfGaussian',
    'Rosenbrock',
]