<img src="https://raw.githubusercontent.com/bambinos/bambi/main/docs/logos/RGB/Bambi_logo.png" width=200>

[![PyPi version](https://badge.fury.io/py/bambi.svg)](https://badge.fury.io/py/bambi)
[![Build Status](https://github.com/bambinos/bambi/actions/workflows/test.yml/badge.svg)](https://github.com/bambinos/bambi/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/bambinos/bambi/branch/master/graph/badge.svg?token=ZqH0KCLKAE)](https://codecov.io/gh/bambinos/bambi)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

# Bambi: Simple Bayesian Modeling in Python

**Bambi is a Python library that simplifies Bayesian model building, making it easy to fit and analyze complex models.**

## Key Features

*   **Intuitive Interface:**  Build Bayesian models with an easy-to-use syntax.
*   **Built on PyMC:** Leverages the powerful PyMC probabilistic programming framework.
*   **Mixed-Effects Model Ready:** Designed specifically for mixed-effects models commonly used in social sciences.
*   **Seamless Integration:** Integrates with ArviZ for model diagnostics, and visualization.
*   **Flexible Family Specification:** Supports common distributions (Gaussian, Bernoulli, etc.) with easy customization.

## What is Bambi?

Bambi is a high-level Bayesian model-building interface designed to make fitting and analyzing Bayesian models straightforward, particularly for researchers and analysts working in fields like the social sciences. It simplifies the process of building and interpreting complex statistical models by abstracting away the complexities of the underlying PyMC framework.

## Getting Started

### Installation

Bambi is easy to install using pip:

```bash
pip install bambi
```

Ensure you have Python 3.11+ and the necessary dependencies installed (NumPy, Pandas, PyMC, ArviZ, and formulae).

### Basic Usage

Here's a quick example of fitting a linear regression model:

```python
import bambi as bmb
import arviz as az
import pandas as pd

# Load your data (replace with your actual data loading)
data = bmb.load_data("sleepstudy")

# Build the model
model = bmb.Model('Reaction ~ Days', data)

# Fit the model
results = model.fit(draws=1000)

# Analyze results
az.summary(results)
az.plot_trace(results)
```

For a deeper dive, explore the [Quickstart](https://github.com/bambinos/bambi#quickstart) and the example notebooks in the [Examples](https://bambinos.github.io/bambi/notebooks/) webpage.

## Examples

Bambi makes it simple to model a variety of regression problems, including linear and logistic regression.

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
```

```python
model = bmb.Model("g['Yes'] ~ x1 + x2", data, family="bernoulli")
fitted = model.fit()
```

## Documentation and Resources

*   [Official Documentation](https://bambinos.github.io/bambi/index.html)
*   [Quickstart](https://github.com/bambinos/bambi#quickstart)
*   [Examples](https://bambinos.github.io/bambi/notebooks/)

## Contributing

Bambi welcomes contributions!  See the [Contributing](https://github.com/bambinos/bambi/blob/main/CONTRIBUTING.md) guidelines.

## Community

*   [Contributors](https://github.com/bambinos/bambi/graphs/contributors)
*   [Code of Conduct](https://github.com/bambinos/bambi/blob/main/CODE_OF_CONDUCT.md)

## Support

*   [Donations](https://numfocus.org/donate-to-pymc) (to PyMC, the project Bambi is built upon)

## License

Bambi is released under the [MIT License](https://github.com/bambinos/bambi/blob/main/LICENSE).

[Back to Top](#bambi-simple-bayesian-modeling-in-python) - [View the Source Code on GitHub](https://github.com/bambinos/bambi)