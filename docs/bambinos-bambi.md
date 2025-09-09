<img src="https://raw.githubusercontent.com/bambinos/bambi/main/docs/logos/RGB/Bambi_logo.png" width=200>

[![PyPi version](https://badge.fury.io/py/bambi.svg)](https://badge.fury.io/py/bambi)
[![Build Status](https://github.com/bambinos/bambi/actions/workflows/test.yml/badge.svg)](https://github.com/bambinos/bambi/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/bambinos/bambi/branch/master/graph/badge.svg?token=ZqH0KCLKAE)](https://codecov.io/gh/bambinos/bambi)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

# Bambi: Bayesian Model Building in Python

**Bambi is a user-friendly Python library that simplifies the process of building and fitting Bayesian statistical models, especially mixed-effects models.**  Find the original repository [here](https://github.com/bambinos/bambi).

## Key Features

*   **Intuitive Interface:** Easily build and specify Bayesian models using a high-level, user-friendly interface.
*   **Built on PyMC:** Leverages the power and flexibility of the PyMC probabilistic programming framework.
*   **Mixed-Effects Models:** Designed for easy fitting of mixed-effects models commonly used in social sciences.
*   **Flexible Model Specification:** Supports a wide range of model types and customizations.
*   **Integration with ArviZ:** Seamlessly integrates with ArviZ for model diagnostics, visualization, and summary statistics.
*   **Simplified Workflow:** Streamlines the Bayesian modeling workflow, from model creation to interpretation.

## Overview

Bambi is a Python library designed to make Bayesian model building accessible and straightforward. Built on top of the powerful [PyMC](https://github.com/pymc-devs/pymc) probabilistic programming framework, Bambi provides a high-level interface that simplifies the creation and fitting of complex statistical models. It is particularly well-suited for researchers in the social sciences who frequently use mixed-effects models. Bambi offers a more intuitive approach to Bayesian modeling compared to directly using PyMC, allowing users to focus on their research questions rather than the complexities of the underlying framework.

## Installation

Bambi requires Python 3.10 or higher. We recommend installing Python and essential numerical libraries through the [Anaconda Distribution](https://www.anaconda.com/products/individual#Downloads) for ease of setup.

To install Bambi using pip:

```bash
pip install bambi
```

For the latest development version:

```bash
pip install git+https://github.com/bambinos/bambi.git
```

### Dependencies

Bambi relies on the following libraries, which should be automatically installed with Bambi:

*   ArviZ
*   formulae
*   NumPy
*   pandas
*   PyMC

## Examples

Here are some basic examples demonstrating how to use Bambi:

```python
import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd

# Linear regression
data = bmb.load_data("sleepstudy")
model = bmb.Model('Reaction ~ Days', data)
results = model.fit(draws=1000)
az.summary(results)
az.plot_trace(results)
```

```
# Logistic regression
data = pd.DataFrame({
    "g": np.random.choice(["Yes", "No"], size=50),
    "x1": np.random.normal(size=50),
    "x2": np.random.normal(size=50)
})
model = bmb.Model("g['Yes'] ~ x1 + x2", data, family="bernoulli")
fitted = model.fit()
```

For more detailed examples, refer to the [Quickstart](https://github.com/bambinos/bambi#quickstart) and the notebooks in the [Examples](https://bambinos.github.io/bambi/notebooks/) webpage.

## Documentation

Comprehensive documentation is available at the [official docs](https://bambinos.github.io/bambi/index.html).

## Citation

If you use Bambi, please cite the following paper:

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

## Contributing

Bambi welcomes contributions from the community. See the [Contributing](https://github.com/bambinos/bambi/blob/main/CONTRIBUTING.md) guide for more details.

## Contributors

See the [GitHub contributor](https://github.com/bambinos/bambi/graphs/contributors) page for a list of contributors.

## Donations

Support the development of PyMC, a sister project, through [donations](https://numfocus.org/donate-to-pymc).

## Code of Conduct

Bambi is committed to maintaining a positive community. Please review the [Code of Conduct](https://github.com/bambinos/bambi/blob/main/CODE_OF_CONDUCT.md).

## License

[MIT License](https://github.com/bambinos/bambi/blob/main/LICENSE)