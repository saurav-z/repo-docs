<img src="https://raw.githubusercontent.com/bambinos/bambi/main/docs/logos/RGB/Bambi_logo.png" width=200>

[![PyPi version](https://badge.fury.io/py/bambi.svg)](https://badge.fury.io/py/bambi)
[![Build Status](https://github.com/bambinos/bambi/actions/workflows/test.yml/badge.svg)](https://github.com/bambinos/bambi/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/bambinos/bambi/branch/master/graph/badge.svg?token=ZqH0KCLKAE)](https://codecov.io/gh/bambinos/bambi)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

# Bambi: Simple Bayesian Model Building in Python

**Bambi simplifies Bayesian model building in Python, making it easy to analyze data with a powerful and flexible framework.**  [Visit the original repository](https://github.com/bambinos/bambi)

## Key Features

*   **Intuitive Interface:** Built on top of PyMC, Bambi provides a high-level interface for specifying and fitting Bayesian models with ease.
*   **Mixed-Effects Model Support:** Effortlessly fit mixed-effects models, common in social sciences and other fields.
*   **Flexible Modeling:**  Supports a wide range of models, including linear and logistic regression, with options for customization.
*   **Integration with ArviZ:** Seamlessly integrates with ArviZ for model diagnostics, summary, and visualization.
*   **Easy Installation:**  Install Bambi with a single `pip` command.
*   **Comprehensive Documentation:** Extensive documentation and examples to get you started quickly.

## What is Bambi?

Bambi is a Python library designed to make Bayesian model building accessible and straightforward. Built upon the robust PyMC probabilistic programming framework, Bambi offers a user-friendly interface for specifying, fitting, and analyzing a wide variety of models. It's particularly well-suited for researchers in the social sciences and related fields who frequently work with mixed-effects models.

## Installation

Install Bambi using pip:

```bash
pip install bambi
```

or to get the latest version from GitHub:

```bash
pip install git+https://github.com/bambinos/bambi.git
```

Ensure you have a working Python environment (3.11+) with the necessary dependencies like NumPy, pandas, and PyMC. The [Anaconda Distribution](https://www.anaconda.com/products/individual#Downloads) is recommended for easy setup.

## Examples

### Linear Regression

```python
import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd

# Load the sleepstudy dataset
data = bmb.load_data("sleepstudy")

# Initialize and fit a model
model = bmb.Model('Reaction ~ Days', data)
results = model.fit(draws=1000)

# Summarize and visualize the results
az.summary(results)
az.plot_trace(results)
```
### Logistic Regression
```python
import pandas as pd
import numpy as np
import bambi as bmb

# Simulate some data
data = pd.DataFrame({
    "g": np.random.choice(["Yes", "No"], size=50),
    "x1": np.random.normal(size=50),
    "x2": np.random.normal(size=50)
})

# Fit a logistic regression model
model = bmb.Model("g['Yes'] ~ x1 + x2", data, family="bernoulli")
fitted = model.fit()
```

For more detailed examples and guidance, explore the [Quickstart](https://github.com/bambinos/bambi#quickstart) and the notebooks in the [Examples](https://bambinos.github.io/bambi/notebooks/) section.

## Documentation

Find detailed documentation at the [official Bambi documentation](https://bambinos.github.io/bambi/index.html).

## Contributing

Bambi welcomes contributions! Review the [Contributing guidelines](https://github.com/bambinos/bambi/blob/main/CONTRIBUTING.md) for details on how to contribute.

## Citation

If you use Bambi in your research, please cite the following:

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

*   **Code of Conduct:** Bambi is committed to fostering a welcoming and inclusive community.  Please review the [Code of Conduct](https://github.com/bambinos/bambi/blob/main/CODE_OF_CONDUCT.md).
*   **Contributors:** See the list of contributors on the [GitHub contributor](https://github.com/bambinos/bambi/graphs/contributors) page.
*   **Donations:** Consider supporting PyMC, Bambi's sister project, by making a [donation](https://numfocus.org/donate-to-pymc).

## License

Bambi is licensed under the [MIT License](https://github.com/bambinos/bambi/blob/main/LICENSE).