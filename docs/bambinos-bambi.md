<img src="https://raw.githubusercontent.com/bambinos/bambi/main/docs/logos/RGB/Bambi_logo.png" width=200>

[![PyPi version](https://badge.fury.io/py/bambi.svg)](https://badge.fury.io/py/bambi)
[![Build Status](https://github.com/bambinos/bambi/actions/workflows/test.yml/badge.svg)](https://github.com/bambinos/bambi/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/bambinos/bambi/branch/master/graph/badge.svg?token=ZqH0KCLKAE)](https://codecov.io/gh/bambinos/bambi)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

# Bambi: Bayesian Modeling Made Easy in Python

**Bambi is a user-friendly Python library that simplifies Bayesian model building, particularly for mixed-effects models, built on top of PyMC.**  [(See the original repository here)](https://github.com/bambinos/bambi)

## Key Features

*   **Intuitive Interface:** Easily define and fit Bayesian models using a clear and concise syntax.
*   **Built on PyMC:** Leverages the power and flexibility of the PyMC probabilistic programming framework.
*   **Mixed-Effects Modeling:** Designed for straightforward implementation of mixed-effects models common in social sciences.
*   **Seamless Integration:** Works smoothly with popular Python data analysis and visualization libraries like ArviZ, pandas, and NumPy.
*   **Flexible:** Supports various model families including linear and logistic regression.
*   **Comprehensive Documentation:**  Provides detailed [official docs](https://bambinos.github.io/bambi/index.html) and examples.

## Overview

Bambi simplifies the process of Bayesian model building in Python. It offers a high-level interface built on PyMC, making it easier to fit complex statistical models, especially mixed-effects models frequently used in fields like social sciences. Bambi takes care of the underlying complexities of Bayesian inference, allowing you to focus on your research questions.

## Installation

Bambi requires Python 3.11 or later. We recommend installing Python and core numerical libraries via the [Anaconda Distribution](https://www.anaconda.com/products/individual#Downloads).

Install Bambi using `pip`:

```bash
pip install bambi
```

For the latest development version:

```bash
pip install git+https://github.com/bambinos/bambi.git
```

### Dependencies

Bambi manages dependencies automatically. Required packages include ArviZ, formulae, NumPy, pandas, and PyMC.  These should be installed during the Bambi installation process.

## Examples

```python
import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd

# Linear regression
data = bmb.load_data("sleepstudy")
data.head()
model = bmb.Model('Reaction ~ Days', data)
print(model)
results = model.fit(draws=1000)
az.summary(results)
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

```python
# Logistic regression
data = pd.DataFrame({
    "g": np.random.choice(["Yes", "No"], size=50),
    "x1": np.random.normal(size=50),
    "x2": np.random.normal(size=50)
})
model = bmb.Model("g['Yes'] ~ x1 + x2", data, family="bernoulli")
fitted = model.fit()
```

For more detailed examples, check out the [Quickstart](https://github.com/bambinos/bambi#quickstart) and explore the notebooks on the [Examples](https://bambinos.github.io/bambi/notebooks/) page.

## Documentation

Comprehensive documentation is available at the [official docs](https://bambinos.github.io/bambi/index.html).

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

## Contributing

Contributions are welcome! See the [Contributing](https://github.com/bambinos/bambi/blob/main/CONTRIBUTING.md) guidelines.

For a list of contributors, see the [GitHub contributor](https://github.com/bambinos/bambi/graphs/contributors) page.

## Donations

Support Bambi by donating to our sister project, PyMC:  [Donate to PyMC](https://numfocus.org/donate-to-pymc)

## Code of Conduct

Bambi is committed to a positive community. See the [Code of Conduct](https://github.com/bambinos/bambi/blob/main/CODE_OF_CONDUCT.md).

## License

[MIT License](https://github.com/bambinos/bambi/blob/main/LICENSE)