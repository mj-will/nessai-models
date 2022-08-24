# nessai-models

Models for use with the nested sampling package [`nessai`](https://github.com/mj-will/nessai).

## Included models

* n-dimensional unit Gaussian
* n-dimensional Rosenbrock
* n-dimensional mixture of Gaussians
* Gaussian mixture using data to based on [this example](https://github.com/johnveitch/cpnest/blob/master/examples/gaussianmixture.py) from `cpnest`
* n-dimensional Egg Box based on the version in [Feroz et al. 2008](https://arxiv.org/abs/0809.3437)

## Requirements

`nessai_models` requires:
* `numpy`
* `scipy`
* `nessai>=0.6.0`

## Installation

> We recommend following the [installation instructions for `nessai`](https://github.com/mj-will/nessai#installation) and then installing `nessai_models` since it shares all of its dependencies with `nessai`.

`nessai_models` is not currently available on PyPI but it can be installed using `pip` directly from the repository:

```
pip install git+https://github.com/igr-ml/nessai-models.git
```

## Example usage

Below is an example of using `nessai_models` so configure a 4-dimensional Gaussian and then sample it using `nessai`.

```python
from nessai import FlowSampler
from nessai_models import Gaussian

model = Gaussian(4)
fs = FlowSampler(model, output='example/')
fs.run()
```
