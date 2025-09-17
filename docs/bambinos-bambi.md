<img src="https://raw.githubusercontent.com/bambinos/bambi/main/docs/logos/RGB/Bambi_logo.png" width=200>

[![PyPi version](https://badge.fury.io/py/bambi.svg)](https://badge.fury.io/py/bambi)
[![Build Status](https://github.com/bambinos/bambi/actions/workflows/test.yml/badge.svg)](https://github.com/bambinos/bambi/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/bambinos/bambi/branch/master/graph/badge.svg?token=ZqH0KCLKAE)](https://codecov.io/gh/bambinos/bambi)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

# Bambi: Simplified Bayesian Modeling in Python

**Bambi is a user-friendly Python library designed to simplify Bayesian model-building, especially for mixed-effects models.** ([Original Repository](https://github.com/bambinos/bambi))

## Key Features

*   **Intuitive Interface:** Easily define and fit Bayesian models using a high-level, declarative syntax.
*   **Built on PyMC:** Leverages the power of the PyMC probabilistic programming framework for robust Bayesian inference.
*   **Simplified Mixed-Effects Modeling:** Streamlines the process of building and analyzing mixed-effects models, common in social sciences.
*   **Flexible Model Specification:** Supports a wide range of models, including linear and logistic regression.
*   **Integration with ArviZ:** Seamlessly integrates with ArviZ for comprehensive model diagnostics, visualization, and interpretation.

## Overview

Bambi provides a user-friendly interface for building and analyzing Bayesian models. Built on top of PyMC, it simplifies the creation of complex models, particularly mixed-effects models, by offering a high-level syntax and a focus on ease of use.

## Installation

Bambi requires Python 3.11+ and can be installed using pip:

```bash
pip install bambi
```

Alternatively, install the development version:

```bash
pip install git+https://github.com/bambinos/bambi.git
```

### Dependencies

Bambi relies on the following Python libraries, which are automatically installed during the Bambi installation process:

*   ArviZ
*   formulae
*   NumPy
*   pandas
*   PyMC

## Examples

### Linear Regression

This example demonstrates a simple fixed-effects model using the "sleepstudy" dataset:

```python
import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd

# Read in a dataset from the package content
data = bmb.load_data("sleepstudy")

# Initialize the fixed effects only model
model = bmb.Model('Reaction ~ Days', data)

# Get model description
print(model)

# Fit the model using 1000 on each chain
results = model.fit(draws=1000)

# Key summary and diagnostic info on the model parameters
az.summary(results)

# Use ArviZ to plot the results
az.plot_trace(results)
```

### Logistic Regression

This example showcases a logistic regression model using a simulated dataset:

```python
data = pd.DataFrame({
    "g": np.random.choice(["Yes", "No"], size=50),
    "x1": np.random.normal(size=50),
    "x2": np.random.normal(size=50)
})

model = bmb.Model("g['Yes'] ~ x1 + x2", data, family="bernoulli")
fitted = model.fit()
```

## Further Resources

*   **Quickstart Guide:** [Quickstart](https://github.com/bambinos/bambi#quickstart)
*   **Examples:** Explore detailed examples and tutorials on the [Examples](https://bambinos.github.io/bambi/notebooks/) webpage.
*   **Documentation:** Access the complete documentation on the [official docs](https://bambinos.github.io/bambi/index.html)

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

Contributions to Bambi are welcome!  Please see the [Contributing](https://github.com/bambinos/bambi/blob/main/CONTRIBUTING.md) Readme for details.

## Contributors

See the list of contributors on the [GitHub contributor](https://github.com/bambinos/bambi/graphs/contributors) page.

## Donations

Support the development of PyMC, the foundation of Bambi, by [making a donation](https://numfocus.org/donate-to-pymc) to our sister project.

## Code of Conduct

Bambi is committed to fostering a positive and inclusive community.  See the [Code of Conduct](https://github.com/bambinos/bambi/blob/main/CODE_OF_CONDUCT.md).

## License

[MIT License](https://github.com/bambinos/bambi/blob/main/LICENSE)