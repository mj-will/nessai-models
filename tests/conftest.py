"""General configuration for all tests"""
from nessai_models import (
    EggBox,
    Gaussian,
    GaussianMixture,
    GaussianMixtureWithData,
    HalfGaussian,
    LinearSignal,
    Pyramid,
    Rosenbrock,
    SinusoidalSignal,
)
import pytest


all_models = [
    EggBox,
    Gaussian,
    GaussianMixture,
    GaussianMixtureWithData,
    HalfGaussian,
    LinearSignal,
    Pyramid,
    Rosenbrock,
    SinusoidalSignal,
]


@pytest.fixture(params=all_models)
def ModelClass(request):
    """Model classes fixture.

    Contains all implemented models.
    """
    return request.param
