<img src="https://raw.githubusercontent.com/bambinos/bambi/main/docs/logos/RGB/Bambi_logo.png" width=200>

[![PyPi version](https://badge.fury.io/py/bambi.svg)](https://badge.fury.io/py/bambi)
[![Build Status](https://github.com/bambinos/bambi/actions/workflows/test.yml/badge.svg)](https://github.com/bambinos/bambi/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/bambinos/bambi/branch/master/graph/badge.svg?token=ZqH0KCLKAE)](https://codecov.io/gh/bambinos/bambi)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

# Bambi: Bayesian Model Building Made Easy in Python

**Bambi empowers researchers with a user-friendly interface for building and fitting Bayesian statistical models using Python, simplifying complex analyses.** [Explore the Bambi repo on GitHub](https://github.com/bambinos/bambi).

## Key Features of Bambi

*   **Intuitive Interface:** Build Bayesian models with a simple, high-level syntax.
*   **Built on PyMC:** Leverages the power of the PyMC probabilistic programming framework.
*   **Mixed-Effects Modeling:** Easily fit complex mixed-effects models commonly used in the social sciences.
*   **Flexible Model Specification:** Supports various model families and link functions for versatile analysis.
*   **Integration with ArviZ:** Seamlessly integrates with ArviZ for model diagnostics, visualization, and summary statistics.
*   **Easy Installation:** Install with a single `pip` command.
*   **Comprehensive Documentation:** Access detailed documentation and examples to get you started quickly.

## Installation

Bambi requires Python 3.11 or higher. It's recommended to install Python and key numerical libraries using the [Anaconda Distribution](https://www.anaconda.com/products/individual#Downloads).

Install Bambi using pip:

```bash
pip install bambi
```

Or install the latest development version from GitHub:

```bash
pip install git+https://github.com/bambinos/bambi.git
```

### Dependencies

Bambi automatically installs the following dependencies: ArviZ, formulae, NumPy, pandas, and PyMC.

## Examples

Here are a few quick examples of using Bambi:

```python
import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd

# Linear regression
data = bmb.load_data("sleepstudy")
model = bmb.Model('Reaction ~ Days', data)
results = model.fit(draws=1000)
az.summary(results)
az.plot_trace(results)

# Logistic regression
data = pd.DataFrame({
    "g": np.random.choice(["Yes", "No"], size=50),
    "x1": np.random.normal(size=50),
    "x2": np.random.normal(size=50)
})
model = bmb.Model("g['Yes'] ~ x1 + x2", data, family="bernoulli")
fitted = model.fit()
```

For more in-depth examples, consult the [Quickstart](https://github.com/bambinos/bambi#quickstart) and the notebooks in the [Examples](https://bambinos.github.io/bambi/notebooks/) webpage.

## Documentation

Detailed documentation can be found in the [official Bambi documentation](https://bambinos.github.io/bambi/index.html).

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

## Contributing

Bambi welcomes contributions!  See the [Contributing Guidelines](https://github.com/bambinos/bambi/blob/main/CONTRIBUTING.md) for more information.  View a list of contributors on the [GitHub contributor](https://github.com/bambinos/bambi/graphs/contributors) page.

## Donations

Support Bambi by donating to its sister project, PyMC: [Make a donation](https://numfocus.org/donate-to-pymc).

## Code of Conduct

Bambi is committed to fostering a positive community.  See the [Code of Conduct](https://github.com/bambinos/bambi/blob/main/CODE_OF_CONDUCT.md) for details.

## License

[MIT License](https://github.com/bambinos/bambi/blob/main/LICENSE)