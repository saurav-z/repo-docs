<img src="https://raw.githubusercontent.com/bambinos/bambi/main/docs/logos/RGB/Bambi_logo.png" width=200>

[![PyPi version](https://badge.fury.io/py/bambi.svg)](https://badge.fury.io/py/bambi)
[![Build Status](https://github.com/bambinos/bambi/actions/workflows/test.yml/badge.svg)](https://github.com/bambinos/bambi/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/bambinos/bambi/branch/master/graph/badge.svg?token=ZqH0KCLKAE)](https://codecov.io/gh/bambinos/bambi)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

# Bambi: Bayesian Model-Building Made Easy in Python

**Bambi is a user-friendly Python library designed for building and fitting Bayesian statistical models, simplifying complex analyses with a clear and intuitive interface.** [Explore the original repository here](https://github.com/bambinos/bambi).

## Key Features

*   **Simplified Model Specification:** Define models using a formula-based approach, similar to R's syntax, making model building intuitive.
*   **Built on PyMC:** Leverages the power and flexibility of the PyMC probabilistic programming framework for robust Bayesian inference.
*   **Mixed-Effects Models:**  Easily fit mixed-effects models, common in social sciences and other fields, with support for various distributions and link functions.
*   **Integration with ArviZ:** Seamlessly integrates with ArviZ for model diagnostics, visualization, and summarization of results.
*   **Flexible and Extensible:** Allows for customization and extension to suit specific research needs, including support for custom priors and likelihoods.
*   **User-Friendly Interface:** Provides a high-level interface that simplifies the process of specifying, fitting, and interpreting Bayesian models.

## Getting Started

### Installation

Install Bambi using pip:

```bash
pip install bambi
```

Alternatively, install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/bambinos/bambi.git
```

### Dependencies

Bambi depends on the following libraries, which are automatically installed with Bambi:

*   ArviZ
*   formulae
*   NumPy
*   pandas
*   PyMC

### Examples

Bambi makes it easy to specify and fit various types of Bayesian models. Here's a glimpse:

```python
import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd

# Load example data
data = bmb.load_data("sleepstudy")

# Build and fit a linear regression model
model = bmb.Model('Reaction ~ Days', data)
results = model.fit(draws=1000)

# Summarize and visualize results
az.summary(results)
az.plot_trace(results)
```

For more in-depth examples, including logistic regression, refer to the [Quickstart](https://github.com/bambinos/bambi#quickstart) and the [Examples](https://bambinos.github.io/bambi/notebooks/) on the Bambi documentation page.

## Documentation

Comprehensive documentation, including tutorials and API references, is available at the [official Bambi documentation](https://bambinos.github.io/bambi/index.html).

## Contributing

Bambi is a community-driven project and welcomes contributions. See the [Contributing](https://github.com/bambinos/bambi/blob/main/CONTRIBUTING.md) guide for details.

## Citation

If you use Bambi in your research, please cite:

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

## Support and Community

*   **Donations:** Support the development of PyMC, Bambi's core dependency, by donating to [NumFOCUS](https://numfocus.org/donate-to-pymc).
*   **Code of Conduct:** Bambi is committed to a welcoming and inclusive community.  Please review the [Code of Conduct](https://github.com/bambinos/bambi/blob/main/CODE_OF_CONDUCT.md).
*   **License:** [MIT License](https://github.com/bambinos/bambi/blob/main/LICENSE)