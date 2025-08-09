<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/sktime/sktime">
    <img src="https://github.com/sktime/sktime/blob/main/docs/source/images/sktime-logo.svg" alt="Logo" width="175" align="center">
  </a>
  <h3 align="center">sktime</h3>
  <p align="center">
    The Python library for time series analysis, providing a unified interface for a wide range of machine learning tasks!
    <br />
    <a href="https://github.com/sktime/sktime"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://www.sktime.net/en/latest/installation.html">Installation</a>
    ·
    <a href="https://www.sktime.net/en/latest/examples.html">Tutorials</a>
    ·
    <a href="https://www.sktime.net/en/latest/api_reference.html">API Reference</a>
    ·
    <a href="https://www.sktime.net/en/latest/changelog.html">Changelog</a>
  </p>
</div>

## About sktime

sktime is a powerful Python library designed to streamline time series analysis. It offers a unified interface for various time series learning tasks.  This includes, but is not limited to: forecasting, time series classification, clustering, and anomaly detection. Built to integrate seamlessly with the scientific Python ecosystem, it offers  [time series algorithms](https://www.sktime.net/en/stable/estimator_overview.html) and  [scikit-learn] compatible tools.

## Key Features

*   **Unified Interface:** Simplifies time series analysis with a consistent API across various tasks.
*   **Comprehensive Tasks:** Supports forecasting, classification, clustering, anomaly detection, and more.
*   **Algorithm Variety:** Offers a rich collection of time series algorithms.
*   **Model Building:** Provides tools for model building, tuning, and validation, compatible with scikit-learn.
*   **Interoperability:** Integrates with popular libraries like scikit-learn, statsmodels, tsfresh, and more.
*   **Extensibility:** Easy to extend with custom algorithms via clear extension templates.

## What's New

*   **Version 0.38.4 is now available!**  Check out the [release notes](https://www.sktime.net/en/latest/changelog.html) for details.

## Core Functionality

*   **Forecasting:**
    *   Stable and ready to use.
    *   [Tutorial](https://www.sktime.net/en/latest/examples/01_forecasting.html) and [API Reference](https://www.sktime.net/en/latest/api_reference/forecasting.html) available.
    *   [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/forecasting.py) for custom implementations.

*   **Time Series Classification:**
    *   Stable and ready to use.
    *   [Tutorial](https://github.com/sktime/sktime/blob/main/examples/02_classification.ipynb) and [API Reference](https://www.sktime.net/en/latest/api_reference/classification.html).
    *   [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/classification.py).

*   **Time Series Regression:**
    *   Stable and ready to use.
    *   [API Reference](https://www.sktime.net/en/latest/api_reference/regression.html).

*   **Transformations:**
    *   Stable and ready to use.
    *   [Tutorial](https://github.com/sktime/sktime/blob/main/examples/03_transformers.ipynb) and [API Reference](https://www.sktime.net/en/latest/api_reference/transformations.html).
    *   [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/transformer.py).

*   **Detection tasks:**
    *   Maturing
    *   [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/detection.py)

*   **Parameter fitting:**
    *   Maturing
    *   [API Reference](https://www.sktime.net/en/latest/api_reference/param_est.html)
    *   [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/transformer.py)

*   **Time Series Clustering:**
    *   Maturing
    *   [API Reference](https://www.sktime.net/en/latest/api_reference/clustering.html)
    *   [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/clustering.py)

*   **Time Series Distances/Kernels:**
    *   Maturing
    *   [Tutorial](https://github.com/sktime/sktime/blob/main/examples/03_transformers.ipynb) and [API Reference](https://www.sktime.net/en/latest/api_reference/dists_kernels.html).
    *   [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/dist_kern_panel.py)

*   **Time Series Alignment:**
    *   Experimental
    *   [API Reference](https://www.sktime.net/en/latest/api_reference/alignment.html).
    *   [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/alignment.py)

*   **Time Series Splitters:**
    *   Maturing
    *   [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/split.py)

*   **Distributions and simulation:**
    *   Experimental

## Installation

Get started with sktime! See the  [documentation](https://www.sktime.net/en/latest/installation.html) for comprehensive instructions.

*   **Supported OS:** macOS X, Linux, Windows 8.1 or higher.
*   **Python Versions:** 3.8, 3.9, 3.10, 3.11, and 3.12 (64-bit only).
*   **Package Managers:**  `pip` and `conda` (via `conda-forge`).

```bash
# Install using pip
pip install sktime
# Install with all extras (for all functionalities)
pip install sktime[all_extras]

# Install using conda
conda install -c conda-forge sktime
# Install with all extras (for all functionalities)
conda install -c conda-forge sktime-all-extras
```

## Quickstart

### Forecasting Example

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

### Time Series Classification Example

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

## Get Involved

Join the sktime community and contribute!  All contributions are welcome!

*   **[Contribute](https://www.sktime.net/en/latest/get_involved/contributing.html):**  Learn how to contribute.
*   **[Mentoring](https://github.com/sktime/mentoring):**  Apply for our mentoring program.
*   **[Meetings](https://calendar.google.com/calendar/u/0/embed?src=sktime.toolbox@gmail.com&ctz=UTC):**  Join discussions, tutorials, workshops, and sprints.
*   **[Developer Guides](https://www.sktime.net/en/latest/developer_guide.html):**  Learn to develop sktime's codebase.
*   **[Enhancement Proposals](https://github.com/sktime/enhancement-proposals):** Design new features.
*   **[Contributors](https://github.com/sktime/sktime/blob/main/CONTRIBUTORS.md):** See who's contributing.
*   **[Roles](https://www.sktime.net/en/latest/about/team.html):** Explore community roles.
*   **[Donate](https://opencollective.com/sktime):** Support sktime's development.
*   **[Governance](https://www.sktime.net/en/latest/get_involved/governance.html):**  Understand how decisions are made.

## Project Vision

sktime aims to be:

*   **Community-Driven:** Developed by a friendly, collaborative community.
*   **Task-Oriented:** Providing the right tools for specific learning problems.
*   **Ecosystem-Integrated:** Interoperable with scikit-learn, statsmodels, and other libraries.
*   **Model Composition:** Featuring rich model composition and reduction functionality.
*   **User-Friendly:** Based on modern object-oriented design for data science.
*   **Fair and Robust:** Focused on fair model assessment and benchmarking.
*   **Extensible:** Easy to extend with custom algorithms via templates.

## Contributors

Thank you to all our community members for their contributions.

[![Contributors](https://opencollective.com/sktime/contributors.svg?width=600&button=false)](https://github.com/sktime/sktime/graphs/contributors)

[Back to top](https://github.com/sktime/sktime)