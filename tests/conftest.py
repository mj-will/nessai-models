"""General configuration for all tests"""

from nessai_models import (
    Brewer,
    EggBox,
    Gaussian,
    GaussianMixture,
    GaussianMixtureWithData,
    HalfGaussian,
    LinearSignal,
    MixtureOfDistributions,
    Pyramid,
    Rosenbrock,
    SinusoidalSignal,
    SlabSpike,
)
import pytest


all_models = [
    Brewer,
    EggBox,
    Gaussian,
    GaussianMixture,
    GaussianMixtureWithData,
    HalfGaussian,
    LinearSignal,
    MixtureOfDistributions,
    Pyramid,
    Rosenbrock,
    SinusoidalSignal,
    SlabSpike,
]


@pytest.fixture(params=all_models)
def ModelClass(request):
    """Model classes fixture.

    Contains all implemented models.
    """
    return request.param
