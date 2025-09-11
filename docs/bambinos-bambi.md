<img src="https://raw.githubusercontent.com/bambinos/bambi/main/docs/logos/RGB/Bambi_logo.png" width=200>

[![PyPi version](https://badge.fury.io/py/bambi.svg)](https://badge.fury.io/py/bambi)
[![Build Status](https://github.com/bambinos/bambi/actions/workflows/test.yml/badge.svg)](https://github.com/bambinos/bambi/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/bambinos/bambi/branch/master/graph/badge.svg?token=ZqH0KCLKAE)](https://codecov.io/gh/bambinos/bambi)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

# Bambi: Bayesian Model Building in Python

**Bambi simplifies Bayesian model building, making it easy to fit and analyze mixed-effects models using the power of PyMC.**

## Key Features

*   **User-Friendly Interface:** A high-level interface designed for intuitive model specification.
*   **Bayesian Approach:** Leverage the advantages of Bayesian statistical methods for robust inference.
*   **Mixed-Effects Model Support:** Easily fit complex models common in social sciences and other fields.
*   **Built on PyMC:** Integrates seamlessly with the powerful PyMC probabilistic programming framework.
*   **Integration with ArviZ:** Supports easy visualization and diagnostics of model results using ArviZ.
*   **Flexible and Extensible:** Easily customize models and integrate with other Python tools.

## Overview

Bambi is a Python library that provides a user-friendly interface for building and fitting Bayesian statistical models. Built on top of [PyMC](https://github.com/pymc-devs/pymc), Bambi simplifies the process of specifying and analyzing models, especially mixed-effects models, making Bayesian analysis more accessible.  

## Installation

Bambi requires Python 3.10+ and can be installed using `pip`:

```bash
pip install bambi
```

For the latest development version:

```bash
pip install git+https://github.com/bambinos/bambi.git
```

Dependencies are handled automatically during installation.

## Examples

### Linear Regression

```python
import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd

# Load data
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

## Resources

*   **[Quickstart](https://github.com/bambinos/bambi#quickstart)**
*   **[Examples](https://bambinos.github.io/bambi/notebooks/)**
*   **[Documentation](https://bambinos.github.io/bambi/index.html)**

## Contributing

Bambi welcomes contributions! See the [CONTRIBUTING.md](https://github.com/bambinos/bambi/blob/main/CONTRIBUTING.md) file for details.

*   [GitHub Contributors](https://github.com/bambinos/bambi/graphs/contributors)

## Support

*   **Donations:** Support PyMC, the project Bambi relies on, by [making a donation](https://numfocus.org/donate-to-pymc).
*   **Code of Conduct:** Bambi is committed to a positive community. See the [Code of Conduct](https://github.com/bambinos/bambi/blob/main/CODE_OF_CONDUCT.md).

## Citation

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

## License

[MIT License](https://github.com/bambinos/bambi/blob/main/LICENSE)

**[Go to the Bambi Repository](https://github.com/bambinos/bambi) to explore further!**