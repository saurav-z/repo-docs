<img src="https://raw.githubusercontent.com/bambinos/bambi/main/docs/logos/RGB/Bambi_logo.png" width=200>

[![PyPi version](https://badge.fury.io/py/bambi.svg)](https://badge.fury.io/py/bambi)
[![Build Status](https://github.com/bambinos/bambi/actions/workflows/test.yml/badge.svg)](https://github.com/bambinos/bambi/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/bambinos/bambi/branch/master/graph/badge.svg?token=ZqH0KCLKAE)](https://codecov.io/gh/bambinos/bambi)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

# Bambi: Bayesian Model Building Made Easy in Python

**Bambi simplifies Bayesian model building, providing an intuitive interface for fitting mixed-effects models using the power of PyMC.**  Learn more and contribute at the [Bambi GitHub repository](https://github.com/bambinos/bambi).

## Key Features

*   **Simplified Bayesian Modeling:** Build and fit Bayesian models with a high-level, user-friendly interface.
*   **Built on PyMC:** Leverages the robust [PyMC](https://github.com/pymc-devs/pymc) probabilistic programming framework.
*   **Mixed-Effects Model Ready:**  Designed for ease of use with mixed-effects models, common in social sciences.
*   **Integration with ArviZ:** Seamlessly integrates with ArviZ for model diagnostics, visualization, and interpretation.
*   **Flexible and Extensible:** Easily handles various model families (e.g., Gaussian, Bernoulli) and link functions.
*   **Comprehensive Documentation:**  Access in-depth information through the [official docs](https://bambinos.github.io/bambi/index.html) and examples.

## Installation

Bambi is easy to install using pip:

```bash
pip install bambi
```

Alternatively, install the latest development version:

```bash
pip install git+https://github.com/bambinos/bambi.git
```

### Dependencies

Bambi relies on the following libraries, which are automatically installed with Bambi:

*   ArviZ
*   formulae
*   NumPy
*   pandas
*   PyMC

## Examples

Here's how to get started with Bambi:

```python
import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd

# Linear regression example
data = bmb.load_data("sleepstudy")
model = bmb.Model('Reaction ~ Days', data)
results = model.fit(draws=1000)
az.summary(results)
az.plot_trace(results)

# Logistic regression example
data = pd.DataFrame({
    "g": np.random.choice(["Yes", "No"], size=50),
    "x1": np.random.normal(size=50),
    "x2": np.random.normal(size=50)
})
model = bmb.Model("g['Yes'] ~ x1 + x2", data, family="bernoulli")
fitted = model.fit()
```

For more detailed examples, including a Quickstart guide and a variety of notebooks, visit the [Bambi Examples](https://bambinos.github.io/bambi/notebooks/) page.

## Documentation

Explore the full potential of Bambi with our comprehensive [official documentation](https://bambinos.github.io/bambi/index.html).

## Contributing

Bambi thrives on community contributions! Check out the [Contributing](https://github.com/bambinos/bambi/blob/main/CONTRIBUTING.md) guide for how to get involved.  See the [GitHub contributor](https://github.com/bambinos/bambi/graphs/contributors) page for a list of contributors.

## Donations

Support the development of PyMC, a key dependency of Bambi, by [making a donation](https://numfocus.org/donate-to-pymc).

## Code of Conduct

Bambi is committed to a welcoming and respectful community. Please review our [Code of Conduct](https://github.com/bambinos/bambi/blob/main/CODE_OF_CONDUCT.md).

## License

[MIT License](https://github.com/bambinos/bambi/blob/main/LICENSE)

## Citation

If you use Bambi in your research, please cite the following paper:

```bibtex
@article{Capretto2022,
 title={Bambi: A Simple Interface for Fitting {Bayesian} Linear Models in {Python}},
 volume={103},
 url={https://www.jstatsoft.org/index.php/jss/article/view/v103i15},
 doi={10.18637/jss.v103.i15},
 number={15},
 journal={Journal of Statistical Software},
 author={Capretto, Tom\'{a}s and Piho, Camen and Kumar, Ravin and Westfall, Jacob and Yarkoni, Tal and Martin, Osvaldo A},
 year={2022},
 pages={1–29}
}
```