"""General configuration for all tests"""
from nessai_models import (
    EggBox,
    Gaussian,
    GaussianMixture,
    GaussianMixtureWithData,
    HalfGaussian,
    Pyramid,
    Rosenbrock,
)
import pytest


all_models = [
    EggBox,
    Gaussian,
    GaussianMixture,
    GaussianMixtureWithData,
    HalfGaussian,
    Pyramid,
    Rosenbrock,
]


@pytest.fixture(params=all_models)
def ModelClass(request):
    """Model classes fixture.

    Contains all implemented models.
    """
    return request.param
