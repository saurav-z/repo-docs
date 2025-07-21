<!-- Improved README for sktime -->

# sktime: The Python Library for Time Series Analysis

<p align="center">
    <a href="https://www.sktime.net"><img src="https://github.com/sktime/sktime/blob/main/docs/source/images/sktime-logo.svg" width="175" alt="sktime logo" /></a>
</p>

**Analyze time series data with ease using sktime, a unified and versatile Python library.**  Version 0.38.4 is now available! ([Release Notes](https://www.sktime.net/en/latest/changelog.html))  Explore a comprehensive toolkit for time series analysis, including forecasting, classification, clustering, anomaly detection, and more.

**Key Features:**

*   **Unified Interface:** Consistent API for various time series tasks.
*   **Diverse Algorithms:** Includes dedicated time series algorithms and scikit-learn compatible tools.
*   **Model Building:**  Tools for building, tuning, and validating time series models.
*   **Interoperability:** Interfaces with popular libraries such as scikit-learn, statsmodels, and others.
*   **Extensibility:**  Easy-to-use extension templates for adding custom algorithms.

[**View the sktime repository on GitHub**](https://github.com/sktime/sktime)

| Resources | Links |
|---|---|
| **Documentation** | [Documentation](https://www.sktime.net/en/stable/users.html) |
| **Tutorials** | [Tutorials](https://www.sktime.net/en/stable/examples.html) |
| **Release Notes** | [Release Notes](https://www.sktime.net/en/stable/changelog.html) |
| **Open Source** | [![BSD 3-clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/sktime/sktime/blob/main/LICENSE) [![GC.OS Sponsored](https://img.shields.io/badge/GC.OS-Sponsored%20Project-orange.svg?style=flat&colorA=0eac92&colorB=2077b4)](https://gc-os-ai.github.io/) |
| **Community Tutorials** | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sktime/sktime/main?filepath=examples) [![!youtube](https://img.shields.io/static/v1?logo=youtube&label=YouTube&message=tutorials&color=red)](https://www.youtube.com/playlist?list=PLKs3UgGjlWHqNzu0LEOeLKvnjvvest2d0) |
| **Community** | [![!discord](https://img.shields.io/static/v1?logo=discord&label=discord&message=chat&color=lightgreen)](https://discord.com/invite/54ACzaFsn7) [![!slack](https://img.shields.io/static/v1?logo=linkedin&label=LinkedIn&message=news&color=lightblue)](https://www.linkedin.com/company/scikit-time/)  |
| **CI/CD** | [![github-actions](https://img.shields.io/github/actions/workflow/status/sktime/sktime/wheels.yml?logo=github)](https://github.com/sktime/sktime/actions/workflows/wheels.yml) [![readthedocs](https://img.shields.io/readthedocs/sktime?logo=readthedocs)](https://www.sktime.net/en/latest/?badge=latest) [![platform](https://img.shields.io/conda/pn/conda-forge/sktime)](https://github.com/sktime/sktime) |
| **Code** |  [![!pypi](https://img.shields.io/pypi/v/sktime?color=orange)](https://pypi.org/project/sktime/) [![!conda](https://img.shields.io/conda/vn/conda-forge/sktime)](https://anaconda.org/conda-forge/sktime) [![!python-versions](https://img.shields.io/pypi/pyversions/sktime)](https://www.python.org/) [![!black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)  |
| **Downloads** | ![PyPI - Downloads](https://img.shields.io/pypi/dw/sktime) ![PyPI - Downloads](https://img.shields.io/pypi/dm/sktime) [![Downloads](https://static.pepy.tech/personalized-badge/sktime?period=total&units=international_system&left_color=grey&right_color=blue&left_text=cumulative%20(pypi))](https://pepy.tech/project/sktime) |
| **Citation** | [![!zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.3749000.svg)](https://doi.org/10.5281/zenodo.3749000) |

## Explore the Documentation

*   **[Tutorials](https://www.sktime.net/en/latest/tutorials.html):** Get started with sktime.
*   **[Binder Notebooks](https://mybinder.org/v2/gh/sktime/sktime/main?filepath=examples):** Interactive examples in your browser.
*   **[Examples](https://www.sktime.net/en/latest/examples.html):** Practical use cases and feature demonstrations.
*   **[API Reference](https://www.sktime.net/en/latest/api_reference.html):** Detailed API documentation.
*   **[Changelog](https://www.sktime.net/en/latest/changelog.html):** Track changes and version history.

## Get Involved & Ask Questions

Your contributions and questions are highly encouraged! Find help and participate in discussions via:

*   **Bug Reports & Feature Requests:** [GitHub Issue Tracker](https://github.com/sktime/sktime/issues)
*   **Usage Questions & General Discussion:** [GitHub Discussions](https://github.com/sktime/sktime/discussions) & [Stack Overflow](https://stackoverflow.com/questions/tagged/sktime)
*   **Contribution & Development:** `dev-chat` channel & [Discord](https://discord.com/invite/54ACzaFsn7)

## sktime Features

sktime offers a unified interface for various time series learning tasks, including:

*   **Forecasting:**  Predict future values.
*   **Time Series Classification:** Categorize time series data.
*   **Time Series Regression:** Predict continuous values from time series.
*   **Transformations:** Preprocessing and feature engineering.
*   **Detection Tasks:**  Identify anomalies and change points.
*   **Time Series Clustering:** Group similar time series.

**Key Modules:**

| Module                         | Status   | Links                                                                                                       |
| ------------------------------ | -------- | ----------------------------------------------------------------------------------------------------------- |
| **Forecasting**                | Stable   | [Tutorial](https://www.sktime.net/en/latest/examples/01_forecasting.html) · [API Reference](https://www.sktime.net/en/latest/api_reference/forecasting.html)  |
| **Time Series Classification** | Stable   | [Tutorial](https://github.com/sktime/sktime/blob/main/examples/02_classification.ipynb) · [API Reference](https://www.sktime.net/en/latest/api_reference/classification.html)  |
| **Time Series Regression**     | Stable   | [API Reference](https://www.sktime.net/en/latest/api_reference/regression.html)                      |
| **Transformations**            | Stable   | [Tutorial](https://github.com/sktime/sktime/blob/main/examples/03_transformers.ipynb) · [API Reference](https://www.sktime.net/en/latest/api_reference/transformations.html)   |
| **Detection Tasks**            | Maturing | [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/detection.py)           |
| **Parameter Fitting**           | Maturing | [API Reference](https://www.sktime.net/en/latest/api_reference/param_est.html)                            |
| **Time Series Clustering**     | Maturing | [API Reference](https://www.sktime.net/en/latest/api_reference/clustering.html)                           |
| **Time Series Distances/Kernels** | Maturing | [Tutorial](https://github.com/sktime/sktime/blob/main/examples/03_transformers.ipynb) · [API Reference](https://www.sktime.net/en/latest/api_reference/dists_kernels.html)   |
| **Time Series Alignment**      | Experimental | [API Reference](https://www.sktime.net/en/latest/api_reference/alignment.html)                               |
| **Time Series Splitters**      | Maturing |  |
| **Distributions and simulation** | Experimental |  |

## Install sktime

Ensure you have the prerequisites and choose your preferred method:

*   **Operating System:** macOS X, Linux, Windows 8.1 or higher
*   **Python:** 3.8, 3.9, 3.10, 3.11, and 3.12 (64-bit only)

**Package Managers:**  [pip](https://pip.pypa.io/en/stable/) or [conda](https://docs.conda.io/en/latest/) (via `conda-forge`)

### Installing with pip:

```bash
pip install sktime
```

For all extras:

```bash
pip install sktime[all_extras]
```

For curated sets of soft dependencies for specific learning tasks:
```bash
pip install sktime[forecasting]  # for selected forecasting dependencies
pip install sktime[forecasting,transformations]  # forecasters and transformers
```

### Installing with conda:

```bash
conda install -c conda-forge sktime
```

For all extras:

```bash
conda install -c conda-forge sktime-all-extras
```

## Quickstart Examples

### Forecasting

```python
from sktime.datasets import load_airline
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.theta import ThetaForecaster
from sktime.split import temporal_train_test_split
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error

y = load_airline()
y_train, y_test = temporal_train_test_split(y)
fh = ForecastingHorizon(y_test.index, is_relative=False)
forecaster = ThetaForecaster(sp=12)  # monthly seasonal periodicity
forecaster.fit(y_train)
y_pred = forecaster.predict(fh)
mean_absolute_percentage_error(y_test, y_pred)
>>> 0.08661467738190656
```

### Time Series Classification

```python
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.datasets import load_arrow_head
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_arrow_head()
X_train, X_test, y_train, y_test = train_test_split(X, y)
classifier = TimeSeriesForestClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy_score(y_test, y_pred)
>>> 0.8679245283018868
```

## Join the sktime Community

Contribute, mentor others, attend meetups, and collaborate:

*   **[Contribute](https://www.sktime.net/en/latest/get_involved/contributing.html):** Contribute to the project.
*   **[Mentoring](https://github.com/sktime/mentoring):** Apply for the mentoring program.
*   **[Meetings](https://calendar.google.com/calendar/u/0/embed?src=sktime.toolbox@gmail.com&ctz=UTC):** Participate in discussions and workshops.
*   **[Developer Guides](https://www.sktime.net/en/latest/developer_guide.html):** Learn how to contribute to the code.
*   **[Contributors](https://github.com/sktime/sktime/blob/main/CONTRIBUTORS.md):** View all contributors.
*   **[Governance](https://www.sktime.net/en/latest/get_involved/governance.html):**  Understand how decisions are made.
*   **[Donate](https://opencollective.com/sktime):** Support the project.

## Hall of Fame

Special thanks to our community for their valuable contributions:

<a href="https://github.com/sktime/sktime/graphs/contributors">
<img src="https://opencollective.com/sktime/contributors.svg?width=600&button=false" alt="Contributors"/>
</a>

## Project Vision

*   **Community-Driven:** Developed by a collaborative community.
*   **Problem-Focused:** Helps users find the right tools for their time series tasks.
*   **Interoperable:** Integrates with popular libraries like scikit-learn.
*   **Model Composition & Reduction:** Build pipelines, tune models, and extract features.
*   **Clean Syntax:** Uses modern object-oriented design principles.
*   **Fair Assessment:** Supports robust model evaluation and benchmarking.
*   **Extensible:** Easy to add custom algorithms.