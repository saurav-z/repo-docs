<img src="https://raw.githubusercontent.com/bambinos/bambi/main/docs/logos/RGB/Bambi_logo.png" width=200>

[![PyPi version](https://badge.fury.io/py/bambi.svg)](https://badge.fury.io/py/bambi)
[![Build Status](https://github.com/bambinos/bambi/actions/workflows/test.yml/badge.svg)](https://github.com/bambinos/bambi/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/bambinos/bambi/branch/master/graph/badge.svg?token=ZqH0KCLKAE)](https://codecov.io/gh/bambinos/bambi)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

# Bambi: Bayesian Model-Building for Python

**Bambi simplifies Bayesian model building in Python, making it easy to fit mixed-effects models for statistical analysis.**

## Key Features

*   **Intuitive Interface:** Bambi offers a high-level, user-friendly interface for defining and fitting Bayesian models.
*   **Built on PyMC:** Leverages the power and flexibility of the PyMC probabilistic programming framework.
*   **Mixed-Effects Models:** Designed specifically for fitting mixed-effects models commonly used in social sciences and related fields.
*   **Easy Installation:** Install Bambi effortlessly using pip.
*   **Integration with ArviZ:** Seamlessly integrates with ArviZ for model diagnostics, visualization, and result interpretation.
*   **Flexible Model Specification:** Supports a wide range of model types, including linear and logistic regression.
*   **Comprehensive Documentation:** Detailed documentation and examples to guide you through the model-building process.

## Overview

Bambi is a Python package that provides a high-level interface for building and fitting Bayesian statistical models. Built on top of the powerful [PyMC](https://github.com/pymc-devs/pymc) framework, Bambi makes it straightforward to specify and analyze complex models, particularly mixed-effects models frequently used in fields like social sciences, psychology, and healthcare.  It simplifies the process of Bayesian model building, allowing researchers and analysts to focus on their research questions rather than getting bogged down in implementation details.

## Installation

Bambi requires Python 3.10+ and can be easily installed using pip:

```bash
pip install bambi
```

For the latest features, you can install directly from GitHub:

```bash
pip install git+https://github.com/bambinos/bambi.git
```

Bambi automatically installs its dependencies, which are listed in `pyproject.toml`.

## Examples

Bambi offers an intuitive syntax to define and fit Bayesian models.

**Linear Regression:**

```python
import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd

# Read in a dataset from the package content
data = bmb.load_data("sleepstudy")

# Initialize the fixed effects only model
model = bmb.Model('Reaction ~ Days', data)

# Fit the model using 1000 on each chain
results = model.fit(draws=1000)

# Key summary and diagnostic info on the model parameters
az.summary(results)

# Use ArviZ to plot the results
az.plot_trace(results)
```

**Logistic Regression:**

```python
data = pd.DataFrame({
    "g": np.random.choice(["Yes", "No"], size=50),
    "x1": np.random.normal(size=50),
    "x2": np.random.normal(size=50)
})

model = bmb.Model("g['Yes'] ~ x1 + x2", data, family="bernoulli")
fitted = model.fit()
```

For detailed examples and guidance, explore the [Quickstart](https://github.com/bambinos/bambi#quickstart) and the notebooks in the [Examples](https://bambinos.github.io/bambi/notebooks/) webpage.

## Documentation

Access the complete Bambi documentation at the [official docs](https://bambinos.github.io/bambi/index.html).

## Contributing

Contributions to Bambi are welcome! See the [Contributing](https://github.com/bambinos/bambi/blob/main/CONTRIBUTING.md) guide for details.  The [GitHub contributor](https://github.com/bambinos/bambi/graphs/contributors) page lists contributors.

## Citation

If you use Bambi, please cite it using the following BibTeX entry:

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

## Donations

Support the development of PyMC, the foundation of Bambi, by donating through [NumFOCUS](https://numfocus.org/donate-to-pymc).

## Code of Conduct

Bambi is committed to fostering a welcoming and inclusive community. Review the [Code of Conduct](https://github.com/bambinos/bambi/blob/main/CODE_OF_CONDUCT.md) for more information.

## License

Bambi is licensed under the [MIT License](https://github.com/bambinos/bambi/blob/main/LICENSE).

[Back to the top](#bambi-bayesian-model-building-for-python)