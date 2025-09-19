<img src="https://raw.githubusercontent.com/bambinos/bambi/main/docs/logos/RGB/Bambi_logo.png" width=200>

[![PyPi version](https://badge.fury.io/py/bambi.svg)](https://badge.fury.io/py/bambi)
[![Build Status](https://github.com/bambinos/bambi/actions/workflows/test.yml/badge.svg)](https://github.com/bambinos/bambi/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/bambinos/bambi/branch/master/graph/badge.svg?token=ZqH0KCLKAE)](https://codecov.io/gh/bambinos/bambi)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

# Bambi: Bayesian Model Building in Python

**Bambi is a user-friendly Python library that simplifies Bayesian model building, making it easy to fit mixed-effects models for various research applications.**

## Key Features

*   **Intuitive Interface:** Build and fit Bayesian models with a simple, high-level API.
*   **Built on PyMC:** Leverages the power of the PyMC probabilistic programming framework.
*   **Mixed-Effects Models:** Designed for easy implementation of mixed-effects models commonly used in social sciences.
*   **Flexible:** Supports a wide range of model families and link functions.
*   **Integration with ArviZ:** Seamlessly integrates with ArviZ for model diagnostics, visualization, and results interpretation.
*   **Easy Installation:** Simple installation via pip.

## What is Bambi?

Bambi is a Python package that provides a high-level interface for building and fitting Bayesian models. It's built on top of the PyMC probabilistic programming framework, and is designed to make it extremely easy to fit mixed-effects models, which are common in social sciences settings, using a Bayesian approach. With Bambi, you can build complex models with minimal code, enabling you to focus on your research questions rather than the intricacies of model implementation.

## Installation

Bambi requires Python 3.11 or higher. We recommend using the Anaconda Distribution.

Install Bambi using pip:

```bash
pip install bambi
```

Or, install the latest development version from GitHub:

```bash
pip install git+https://github.com/bambinos/bambi.git
```

### Dependencies

Bambi relies on ArviZ, formulae, NumPy, pandas, and PyMC. These dependencies are automatically installed when you install Bambi.

## Quickstart

```python
import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd

# Load sample data
data = bmb.load_data("sleepstudy")

# Define and fit the model
model = bmb.Model('Reaction ~ Days', data)
results = model.fit(draws=1000)

# Analyze the results
az.summary(results)
az.plot_trace(results)
```
See the [Quickstart](https://github.com/bambinos/bambi#quickstart) and [Examples](https://bambinos.github.io/bambi/notebooks/) for more detailed usage.

## Examples

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

First, we create and build a Bambi `Model`. Then, the method `model.fit()` tells the sampler to start
running and it returns an `InferenceData` object, which can be passed to several ArviZ functions
such as `az.summary()` to get a summary of the parameters distribution and sample diagnostics or
`az.plot_trace()` to visualize them.

### Logistic Regression

```python
data = pd.DataFrame({
    "g": np.random.choice(["Yes", "No"], size=50),
    "x1": np.random.normal(size=50),
    "x2": np.random.normal(size=50)
})
```

Here we just add the `family` argument set to `"bernoulli"` to tell Bambi we are modelling a binary
response. By default, it uses a logit link. We can also use some syntax sugar to specify which event
we want to model. We just say `g['Yes']` and Bambi will understand we want to model the probability
of a `"Yes"` response. But this notation is not mandatory. If we use `"g ~ x1 + x2"`, Bambi will
pick one of the events to model and will inform us which one it picked.

```python
model = bmb.Model("g['Yes'] ~ x1 + x2", data, family="bernoulli")
fitted = model.fit()
```

After this, we can evaluate the model as before. 

## Documentation

Comprehensive documentation is available at [https://bambinos.github.io/bambi/index.html](https://bambinos.github.io/bambi/index.html).

## Contributing

Bambi welcomes contributions! Please see the [CONTRIBUTING.md](https://github.com/bambinos/bambi/blob/main/CONTRIBUTING.md) file for guidelines.

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

## Community & Support

*   **Code of Conduct:** Bambi is committed to fostering a welcoming and inclusive community. See the [Code of Conduct](https://github.com/bambinos/bambi/blob/main/CODE_OF_CONDUCT.md) for more information.
*   **Contributors:** A list of contributors can be found on the [GitHub contributor](https://github.com/bambinos/bambi/graphs/contributors) page.
*   **Donations:** Support the development of PyMC (a related project) by donating to [NumFOCUS](https://numfocus.org/donate-to-pymc).

## License

Bambi is licensed under the [MIT License](https://github.com/bambinos/bambi/blob/main/LICENSE).

[Back to top](#bambi-bayesian-model-building-in-python) (link back to top of page)
```
Key improvements and SEO considerations:

*   **Clear Headline and Hook:** The title is prominent and includes the main keywords. The one-sentence hook immediately grabs the reader's attention.
*   **Keyword Optimization:** The text includes relevant keywords like "Bayesian," "model building," "mixed-effects models," "Python," and "PyMC."
*   **Well-Structured:** Uses clear headings, bullet points for features, and organized examples for readability.
*   **Concise and Informative:**  Provides essential information without being overly verbose.
*   **Call to Action (Implied):** The examples and quickstart encourage users to try the library.
*   **Internal Linking:** Added "Back to top" and references throughout for better navigation.
*   **Emphasis on Benefits:** Highlights the advantages of using Bambi.
*   **Includes Example Code:** Showcases the library's usability with clear, runnable examples.
*   **Complete Information:** Provides all necessary details (installation, usage, documentation, contributing, license).
*   **Community Focus:** Adds sections about contributing, the code of conduct, and ways to support the project.