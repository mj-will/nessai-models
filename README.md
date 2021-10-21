# nessai-models

Models for use with the nested sampling package [`nessai`](https://github.com/mj-will/nessai).

## Included models

* n-dimensional unit Gaussian
* n-dimensional Rosenbrock
* n-dimensional mixture of Gaussians
* Gaussian mixture using data to based on [this example](https://github.com/johnveitch/cpnest/blob/master/examples/gaussianmixture.py) from `cpnest`

## Requirements

`nessaimodels` requires:
* `numpy`
* `scipy`
* `nessai>=0.3.1`
## Installation

`nessaimodels` is not currently available on PyPI but it can be installed using `pip` directly from the repository:

```
pip install git+https://github.com/igr-ml/nessai-models.git
```

## Example usage

Below is an example of using `nessaimodels` so configure a 4-dimensional Gaussian and then sample it using `nessai`.

```python
from nessai import FlowSampler
from nessaimodels import Gaussian

model = Gaussian(4)
fs = FlowSampler(model, output='example/')
fs.run()
```
