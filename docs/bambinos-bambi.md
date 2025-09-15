<img src="https://raw.githubusercontent.com/bambinos/bambi/main/docs/logos/RGB/Bambi_logo.png" width=200>

[![PyPi version](https://badge.fury.io/py/bambi.svg)](https://badge.fury.io/py/bambi)
[![Build Status](https://github.com/bambinos/bambi/actions/workflows/test.yml/badge.svg)](https://github.com/bambinos/bambi/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/bambinos/bambi/branch/master/graph/badge.svg?token=ZqH0KCLKAE)](https://codecov.io/gh/bambinos/bambi)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

# Bambi: Bayesian Model Building in Python

**Bambi simplifies Bayesian statistical modeling in Python, making it easier than ever to build and analyze sophisticated models.**  ([View on GitHub](https://github.com/bambinos/bambi))

## Key Features

*   **Intuitive Interface:**  Build Bayesian models with a straightforward and user-friendly API.
*   **Built on PyMC:** Leverages the power of the PyMC probabilistic programming framework for robust Bayesian inference.
*   **Mixed-Effects Model Support:** Easily fit mixed-effects models, common in social sciences and other fields.
*   **Flexible Model Specification:** Define models using familiar formula syntax.
*   **Integration with ArviZ:**  Seamlessly integrates with ArviZ for powerful diagnostic and visualization tools.
*   **Easy Installation:** Install Bambi quickly using pip.

## Overview

Bambi is a high-level Python library designed to simplify the process of building and fitting Bayesian statistical models. Built on top of PyMC, Bambi provides an intuitive interface for specifying models, particularly mixed-effects models often used in social sciences. This allows researchers to focus on their research questions rather than getting bogged down in the complexities of Bayesian inference.

## Installation

Bambi requires Python 3.10+ and can be easily installed using pip:

```bash
pip install bambi
```

For the most up-to-date version, install directly from GitHub:

```bash
pip install git+https://github.com/bambinos/bambi.git
```

### Dependencies

Bambi relies on the following libraries, which are automatically installed during the Bambi installation process: ArviZ, formulae, NumPy, pandas, and PyMC.  Dependencies are listed in `pyproject.toml`.

## Examples

Here are basic examples to get you started:

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
```

```python
model = bmb.Model("g['Yes'] ~ x1 + x2", data, family="bernoulli")
fitted = model.fit()
```

### More Examples and Quickstart

For a deeper dive into Bambi's capabilities, consult the [Quickstart](https://github.com/bambinos/bambi#quickstart) and explore the notebooks on the [Examples](https://bambinos.github.io/bambi/notebooks/) page.

## Documentation

Comprehensive documentation is available at the [official Bambi documentation](https://bambinos.github.io/bambi/index.html).

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

## Contributing

Bambi is an open-source project and welcomes contributions from the community. See the [Contributing](https://github.com/bambinos/bambi/blob/main/CONTRIBUTING.md) Readme for more details.

For a list of contributors, please see the [GitHub contributor](https://github.com/bambinos/bambi/graphs/contributors) page.

## Donations

Support the development of PyMC, Bambi's underlying probabilistic programming library, by donating to [NumFOCUS](https://numfocus.org/donate-to-pymc).

## Code of Conduct

Bambi is committed to fostering a positive and inclusive community.  Please review the [Code of Conduct](https://github.com/bambinos/bambi/blob/main/CODE_OF_CONDUCT.md).

## License

Bambi is released under the [MIT License](https://github.com/bambinos/bambi/blob/main/LICENSE).