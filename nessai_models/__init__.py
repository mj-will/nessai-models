# -*- coding: utf-8 -*-
"""
Models for use with the nested sampler \
    `nessai <https://github.com/mj-will/nessai-models>`_.
"""
try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:  # for Python < 3.8
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    pass

from .brewer import Brewer
from .eggbox import EggBox
from .gaussian import Gaussian
from .gaussianmixture import GaussianMixture, GaussianMixtureWithData
from .halfgaussian import HalfGaussian
from .mixture import MixtureOfDistributions
from .pyramid import Pyramid
from .rosenbrock import Rosenbrock
from .signals import LinearSignal, SinusoidalSignal

__all__ = [
    "Brewer",
    "EggBox",
    "Gaussian",
    "GaussianMixture",
    "GaussianMixtureWithData",
    "HalfGaussian",
    "LinearSignal",
    "MixtureOfDistributions",
    "Pyramid",
    "Rosenbrock",
    "SinusoidalSignal",
]
