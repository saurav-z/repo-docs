<img src="https://raw.githubusercontent.com/bambinos/bambi/main/docs/logos/RGB/Bambi_logo.png" width=200>

[![PyPi version](https://badge.fury.io/py/bambi.svg)](https://badge.fury.io/py/bambi)
[![Build Status](https://github.com/bambinos/bambi/actions/workflows/test.yml/badge.svg)](https://github.com/bambinos/bambi/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/bambinos/bambi/branch/master/graph/badge.svg?token=ZqH0KCLKAE)](https://codecov.io/gh/bambinos/bambi)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

# Bambi: Bayesian Modeling Made Easy in Python

**Bambi is a user-friendly Python library that simplifies Bayesian model building, making it easy to analyze data and draw insightful conclusions using the power of Bayesian statistics.**  Learn more and contribute on the [Bambi GitHub Repository](https://github.com/bambinos/bambi).

## Key Features

*   **Intuitive Interface:** Build and fit Bayesian models with a simple and familiar formula syntax.
*   **Built on PyMC:** Leverages the robust probabilistic programming capabilities of PyMC for accurate and reliable results.
*   **Mixed-Effects Models:** Easily fit complex mixed-effects models, common in social sciences and other fields.
*   **Flexible and Customizable:** Offers a wide range of customization options to tailor models to your specific needs.
*   **Integration with ArviZ:** Seamlessly integrates with ArviZ for model diagnostics, visualization, and summary statistics.
*   **Easy Installation:** Install Bambi with a single pip command.

## What is Bambi?

Bambi is a high-level Python library designed to simplify Bayesian model building. It provides a user-friendly interface for specifying and fitting Bayesian models, particularly mixed-effects models, making it easier for researchers and analysts to work with complex data and draw meaningful inferences. Bambi is built on top of the PyMC probabilistic programming framework.

## Installation

Install Bambi easily using pip:

```bash
pip install bambi
```

For the latest development version:

```bash
pip install git+https://github.com/bambinos/bambi.git
```

### Dependencies

Bambi relies on the following libraries, which are automatically installed during the installation process:

*   ArviZ
*   formulae
*   NumPy
*   pandas
*   PyMC

## Examples

### Linear Regression

```python
import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd

# Load the sleepstudy dataset
data = bmb.load_data("sleepstudy")

# Display the first few rows of the data
data.head()

# Initialize and fit the model
model = bmb.Model('Reaction ~ Days', data)
results = model.fit(draws=1000)

# Summarize results and generate plots
print(model)
az.summary(results)
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

## Documentation

Comprehensive documentation is available at the [official Bambi documentation](https://bambinos.github.io/bambi/index.html). Explore the [Quickstart](https://github.com/bambinos/bambi#quickstart) and [Examples](https://bambinos.github.io/bambi/notebooks/) for more in-depth information.

## Contributing

Bambi welcomes contributions from the community. Review the [Contributing](https://github.com/bambinos/bambi/blob/main/CONTRIBUTING.md) guidelines for more information.
Find a list of contributors on the [GitHub contributor](https://github.com/bambinos/bambi/graphs/contributors) page.

## Code of Conduct

Bambi is committed to fostering a positive and inclusive community. Refer to the [Code of Conduct](https://github.com/bambinos/bambi/blob/main/CODE_OF_CONDUCT.md) for details.

## Support & Donations

If you want to support Bambi, consider [making a donation](https://numfocus.org/donate-to-pymc) to PyMC, Bambi's sister project.

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