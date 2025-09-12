<img src="https://raw.githubusercontent.com/bambinos/bambi/main/docs/logos/RGB/Bambi_logo.png" width=200>

[![PyPi version](https://badge.fury.io/py/bambi.svg)](https://badge.fury.io/py/bambi)
[![Build Status](https://github.com/bambinos/bambi/actions/workflows/test.yml/badge.svg)](https://github.com/bambinos/bambi/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/bambinos/bambi/branch/master/graph/badge.svg?token=ZqH0KCLKAE)](https://codecov.io/gh/bambinos/bambi)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

# Bambi: Bayesian Model Building in Python

**Bambi is a user-friendly Python library that simplifies Bayesian model building, making it easy to fit and analyze complex statistical models.**  For more details, visit the [Bambi GitHub Repository](https://github.com/bambinos/bambi).

## Key Features

*   **Intuitive Interface:**  Build and specify Bayesian models with a clear and concise syntax.
*   **Built on PyMC:** Leverages the power of the PyMC probabilistic programming framework for robust Bayesian inference.
*   **Mixed-Effects Models:** Designed for easy fitting of mixed-effects models, ideal for social science and other fields.
*   **Flexible Model Specification:** Define models using familiar formula syntax, supporting a wide range of distributions and link functions.
*   **Integration with ArviZ:** Seamlessly integrates with ArviZ for powerful model diagnostics, visualization, and summarization.
*   **Data Handling:** Works directly with Pandas DataFrames and other common data formats.
*   **Easy Installation:** Simple installation via pip.

## Getting Started

### Installation

Install Bambi with pip:

```bash
pip install bambi
```

### Examples

Here's a basic example of how to use Bambi for linear regression:

```python
import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd

# Load data
data = bmb.load_data("sleepstudy")

# Build the model
model = bmb.Model('Reaction ~ Days', data)

# Fit the model
results = model.fit(draws=1000)

# Summarize the results
az.summary(results)

# Visualize the results
az.plot_trace(results)
```

**For more detailed examples and tutorials, explore the:**
*   [Quickstart](https://github.com/bambinos/bambi#quickstart)
*   [Examples](https://bambinos.github.io/bambi/notebooks/)

## Documentation

Access the complete Bambi documentation for comprehensive information on all features and functionalities:

*   [Official Docs](https://bambinos.github.io/bambi/index.html)

## Contributing

Bambi welcomes contributions from the community.  Please see the [Contributing](https://github.com/bambinos/bambi/blob/main/CONTRIBUTING.md) guidelines.  A list of contributors can be found on the [GitHub contributor](https://github.com/bambinos/bambi/graphs/contributors) page.

## Supporting Bambi

Consider supporting the development of Bambi and its ecosystem:

*   [Donate](https://numfocus.org/donate-to-pymc) to our sister project PyMC.

## Code of Conduct

Bambi is committed to fostering a positive and inclusive community. Please review the [Code of Conduct](https://github.com/bambinos/bambi/blob/main/CODE_OF_CONDUCT.md).

## License

Bambi is released under the [MIT License](https://github.com/bambinos/bambi/blob/main/LICENSE).

## Citation

If you use Bambi, please cite the following:

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