<img src="https://raw.githubusercontent.com/bambinos/bambi/main/docs/logos/RGB/Bambi_logo.png" width=200>

[![PyPi version](https://badge.fury.io/py/bambi.svg)](https://badge.fury.io/py/bambi)
[![Build Status](https://github.com/bambinos/bambi/actions/workflows/test.yml/badge.svg)](https://github.com/bambinos/bambi/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/bambinos/bambi/branch/master/graph/badge.svg?token=ZqH0KCLKAE)](https://codecov.io/gh/bambinos/bambi)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

# Bambi: Simplify Bayesian Modeling in Python

**Bambi is a user-friendly Python library built on PyMC, designed to make Bayesian model building and analysis intuitive and accessible.**

## Key Features

*   **Intuitive Interface:** Build Bayesian models with a high-level, easy-to-understand syntax.
*   **Built on PyMC:** Leverage the power and flexibility of the PyMC probabilistic programming framework.
*   **Mixed-Effects Models:** Easily fit mixed-effects models commonly used in social sciences.
*   **Flexible:** Supports a wide range of model families and link functions.
*   **Integration with ArviZ:** Seamlessly integrates with ArviZ for model diagnostics, visualization, and inference.

## Overview

Bambi provides a simplified interface for building and analyzing Bayesian models. It is built on top of the powerful [PyMC](https://github.com/pymc-devs/pymc) probabilistic programming framework.  Bambi is particularly well-suited for researchers in the social sciences who want to fit mixed-effects models using a Bayesian approach.

## Installation

Bambi requires Python 3.11 or later.

### Recommended Installation with pip

```bash
pip install bambi
```

### Installing from GitHub

For the latest, development version:

```bash
pip install git+https://github.com/bambinos/bambi.git
```

### Dependencies

Bambi automatically installs its dependencies, which are listed in `pyproject.toml`. These include ArviZ, formulae, NumPy, pandas, and PyMC.

## Examples

### Linear Regression

```python
import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd

# Load the example dataset
data = bmb.load_data("sleepstudy")

# Initialize the model
model = bmb.Model('Reaction ~ Days', data)

# Fit the model
results = model.fit(draws=1000)

# Summarize the results
az.summary(results)

# Visualize the results
az.plot_trace(results)
```

### Logistic Regression

```python
data = pd.DataFrame({
    "g": np.random.choice(["Yes", "No"], size=50),
    "x1": np.random.normal(size=50),
    "x2": np.random.normal(size=50)
})

model = bmb.Model("g['Yes'] ~ x1 + x2", data, family="bernoulli")
fitted = model.fit()
```

## Documentation

Comprehensive documentation is available at the [official Bambi documentation](https://bambinos.github.io/bambi/index.html).

## Contributing

Contributions are welcome!  See the [Contributing](https://github.com/bambinos/bambi/blob/main/CONTRIBUTING.md) guidelines.

## Support

*   [Quickstart](https://github.com/bambinos/bambi#quickstart)
*   [Examples](https://bambinos.github.io/bambi/notebooks/)

## Citation

If you use Bambi in your research, please cite it using the following BibTeX entry:

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

## Community

*   [Code of Conduct](https://github.com/bambinos/bambi/blob/main/CODE_OF_CONDUCT.md)
*   [GitHub Contributors](https://github.com/bambinos/bambi/graphs/contributors)

## Donations

Support the development of PyMC (Bambi's core dependency) by donating to [NumFOCUS](https://numfocus.org/donate-to-pymc).

## License

[MIT License](https://github.com/bambinos/bambi/blob/main/LICENSE)

[Back to top](https://github.com/bambinos/bambi)