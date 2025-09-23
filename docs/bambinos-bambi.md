<img src="https://raw.githubusercontent.com/bambinos/bambi/main/docs/logos/RGB/Bambi_logo.png" width=200>

[![PyPi version](https://badge.fury.io/py/bambi.svg)](https://badge.fury.io/py/bambi)
[![Build Status](https://github.com/bambinos/bambi/actions/workflows/test.yml/badge.svg)](https://github.com/bambinos/bambi/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/bambinos/bambi/branch/master/graph/badge.svg?token=ZqH0KCLKAE)](https://codecov.io/gh/bambinos/bambi)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

# Bambi: Bayesian Model Building Made Easy in Python

**Bambi is your go-to Python library for effortlessly building and fitting Bayesian mixed-effects models.**

## Key Features

*   **Intuitive Interface:** Simplifies the process of specifying and fitting Bayesian models.
*   **Built on PyMC:** Leverages the power of the PyMC probabilistic programming framework.
*   **Mixed-Effects Models:** Designed for easy implementation of mixed-effects models, common in social sciences.
*   **Easy Installation:** Install Bambi with a simple `pip install bambi`.
*   **Integration with ArviZ:** Seamlessly integrates with ArviZ for model diagnostics and visualization.
*   **Comprehensive Documentation:** Extensive documentation to guide you through the model-building process.

## Overview

Bambi provides a high-level interface for Bayesian model building in Python, built on top of the powerful [PyMC](https://github.com/pymc-devs/pymc) framework.  It is specifically designed to streamline the creation and analysis of mixed-effects models, often used in the social sciences, making Bayesian modeling accessible and efficient.

## Installation

Bambi requires Python 3.11+ and can be easily installed using pip:

```bash
pip install bambi
```

For the latest development version:

```bash
pip install git+https://github.com/bambinos/bambi.git
```

### Dependencies

Bambi automatically handles its dependencies, which are listed in `pyproject.toml`.  These include ArviZ, formulae, NumPy, pandas, and PyMC.

## Examples

Below are examples of how to build and fit a model:

```python
import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd

# Linear Regression
data = bmb.load_data("sleepstudy")  # Load sample dataset

model = bmb.Model('Reaction ~ Days', data)  # Define the model
results = model.fit(draws=1000)  # Fit the model
az.summary(results)  # Summarize the results
az.plot_trace(results)  # Visualize the trace plots
```

```python
# Logistic Regression
data = pd.DataFrame({
    "g": np.random.choice(["Yes", "No"], size=50),
    "x1": np.random.normal(size=50),
    "x2": np.random.normal(size=50)
})

model = bmb.Model("g['Yes'] ~ x1 + x2", data, family="bernoulli") # Define the model
fitted = model.fit() # Fit the model
```

## Resources

*   **Quickstart:**  [Quickstart](https://github.com/bambinos/bambi#quickstart)
*   **Examples:** [Examples](https://bambinos.github.io/bambi/notebooks/)
*   **Documentation:**  [Official Docs](https://bambinos.github.io/bambi/index.html)

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

Bambi welcomes contributions from the community.  See the [Contributing](https://github.com/bambinos/bambi/blob/main/CONTRIBUTING.md) guidelines for details.  View the list of contributors on the [GitHub contributor](https://github.com/bambinos/bambi/graphs/contributors) page.

## Donations

Support the development of PyMC, Bambi's underlying framework, by donating to [NumFOCUS](https://numfocus.org/donate-to-pymc).

## Code of Conduct

Bambi is committed to maintaining a positive and inclusive community.  Review the [Code of Conduct](https://github.com/bambinos/bambi/blob/main/CODE_OF_CONDUCT.md) for details.

## License

Bambi is licensed under the [MIT License](https://github.com/bambinos/bambi/blob/main/LICENSE).

[Go back to the original repo](https://github.com/bambinos/bambi)