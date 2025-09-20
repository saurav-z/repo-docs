<img src="https://raw.githubusercontent.com/bambinos/bambi/main/docs/logos/RGB/Bambi_logo.png" width=200>

[![PyPi version](https://badge.fury.io/py/bambi.svg)](https://badge.fury.io/py/bambi)
[![Build Status](https://github.com/bambinos/bambi/actions/workflows/test.yml/badge.svg)](https://github.com/bambinos/bambi/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/bambinos/bambi/branch/master/graph/badge.svg?token=ZqH0KCLKAE)](https://codecov.io/gh/bambinos/bambi)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

# Bambi: Simplify Bayesian Modeling in Python

**Bambi is a user-friendly Python library that streamlines the process of building and fitting Bayesian statistical models.**

## Key Features

*   **Intuitive Interface:** Easily define and fit Bayesian models using a high-level, formula-based approach.
*   **Built on PyMC:** Leverages the power and flexibility of the PyMC probabilistic programming framework.
*   **Mixed-Effects Models:** Designed for effortlessly fitting mixed-effects models, common in social sciences.
*   **Integration with ArviZ:** Seamlessly integrates with ArviZ for model diagnostics, exploration, and visualization.
*   **Simple Installation:** Easy to install using `pip`.

## Overview

Bambi provides a user-friendly interface for Bayesian model-building in Python.  It simplifies the process of specifying, fitting, and analyzing Bayesian models, particularly mixed-effects models, making it accessible to researchers and practitioners across various fields. Built on top of the powerful PyMC framework, Bambi enables you to leverage the benefits of Bayesian statistics with ease.

## Installation

Bambi requires Python 3.11+ and recommends the Anaconda Distribution for easy setup of dependencies.

Install Bambi using pip:

```bash
pip install bambi
```

Or install the development version from GitHub:

```bash
pip install git+https://github.com/bambinos/bambi.git
```

### Dependencies

Bambi depends on ArviZ, formulae, NumPy, pandas, and PyMC. These dependencies are handled during installation.

## Examples

Below are examples of implementing linear and logistic regressions using the library.

```python
import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd
```

### Linear Regression

```python
# Read in a dataset from the package content
data = bmb.load_data("sleepstudy")

# See first rows
data.head()
 
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
``` 
   Reaction  Days  Subject
0  249.5600     0      308
1  258.7047     1      308
2  250.8006     2      308
3  321.4398     3      308
4  356.8519     4      308
```
```
       Formula: Reaction ~ Days
        Family: gaussian
          Link: mu = identity
  Observations: 180
        Priors:
    target = mu
        Common-level effects
            Intercept ~ Normal(mu: 298.5079, sigma: 261.0092)
            Days ~ Normal(mu: 0.0, sigma: 48.8915)

        Auxiliary parameters
            sigma ~ HalfStudentT(nu: 4.0, sigma: 56.1721)
```
```
                   mean     sd   hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
Intercept       251.552  6.658  238.513  263.417      0.083    0.059    6491.0    2933.0    1.0
Days             10.437  1.243    8.179   12.793      0.015    0.011    6674.0    3242.0    1.0
Reaction_sigma   47.949  2.550   43.363   52.704      0.035    0.025    5614.0    2974.0    1.0
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

## Further Resources

*   **Quickstart Guide:** Get started with Bambi quickly.
*   **Examples:** Explore various modeling scenarios and learn from practical examples.
*   **Documentation:** Access detailed documentation for Bambi's functions and features.

## Contributing

Bambi welcomes contributions from the community.  See the [Contributing](https://github.com/bambinos/bambi/blob/main/CONTRIBUTING.md) guidelines for more information.

## Get Involved

*   **Contributors:** Explore the project's [GitHub contributor](https://github.com/bambinos/bambi/graphs/contributors) page.
*   **Support PyMC:**  Consider [donating](https://numfocus.org/donate-to-pymc) to PyMC, Bambi's sister project.
*   **Code of Conduct:**  Review the [Code of Conduct](https://github.com/bambinos/bambi/blob/main/CODE_OF_CONDUCT.md) for community guidelines.

## License

Bambi is licensed under the [MIT License](https://github.com/bambinos/bambi/blob/main/LICENSE).

**[Visit the Bambi GitHub Repository](https://github.com/bambinos/bambi) to learn more and get started.**