<!-- SEO-optimized README for sktime -->

# sktime: Your Unified Toolkit for Time Series Analysis

[![sktime Logo](https://github.com/sktime/sktime/blob/main/docs/source/images/sktime-logo.svg?raw=true "sktime Logo")](https://www.sktime.net/)

**sktime simplifies time series analysis by providing a consistent and powerful interface for all your machine learning needs.**

:rocket: **Version 0.38.4 is now available!** [Read the release notes](https://www.sktime.net/en/latest/changelog.html).

sktime is a comprehensive Python library designed for time series analysis, offering a unified interface for various time series learning tasks. It empowers you to build, tune, and validate your time series models with ease.

**Key Features:**

*   **Unified Interface:** Simplify time series analysis with a consistent API.
*   **Multiple Tasks:** Covers forecasting, classification, clustering, anomaly detection, and more.
*   **Extensive Algorithms:** Includes a wide range of time series algorithms.
*   **scikit-learn Compatibility:** Seamless integration with scikit-learn tools.
*   **Model Building & Tuning:** Tools for building, tuning, and validating time series models.
*   **Interoperability:** Interfaces with related libraries like scikit-learn, statsmodels, and others.
*   **Open Source:**  [BSD 3-clause License](https://github.com/sktime/sktime/blob/main/LICENSE)

**[View the sktime Documentation](https://www.sktime.net/en/stable/users.html)**

| Resources                         | Links                                                                                              |
| --------------------------------- | -------------------------------------------------------------------------------------------------- |
| **Documentation**                  | [Tutorials](https://www.sktime.net/en/latest/examples.html) · [Release Notes](https://www.sktime.net/en/latest/changelog.html) |
| **Community**                    | [Discord](https://discord.com/invite/54ACzaFsn7) · [LinkedIn](https://www.linkedin.com/company/scikit-time/) |
| **Code & Downloads**               | [PyPI](https://pypi.org/project/sktime/) · [Conda](https://anaconda.org/conda-forge/sktime)  |
| **Development & CI/CD**           | [GitHub Actions](https://github.com/sktime/sktime/actions/workflows/wheels.yml) · [Read the Docs](https://www.sktime.net/en/latest/?badge=latest) |

## Explore sktime's Resources

*   **[Tutorials](https://www.sktime.net/en/latest/tutorials.html):** Get started with sktime.
*   **[Binder Notebooks](https://mybinder.org/v2/gh/sktime/sktime/main?filepath=examples):** Interactive examples in your browser.
*   **[Examples](https://www.sktime.net/en/latest/examples.html):** Learn how to use sktime features.
*   **[API Reference](https://www.sktime.net/en/latest/api_reference.html):** Detailed API documentation.
*   **[Changelog](https://www.sktime.net/en/latest/changelog.html):** View version history.
*   **[Roadmap](https://www.sktime.net/en/latest/roadmap.html):** Learn about sktime's software and community development plan.

## Ask Questions & Get Involved

We encourage community participation and welcome your feedback!

| Area                                 | Platforms                                           |
| ------------------------------------ | --------------------------------------------------- |
| **Bug Reports**                      | [GitHub Issue Tracker](https://github.com/sktime/sktime/issues)         |
| **Feature Requests & Ideas**           | [GitHub Issue Tracker](https://github.com/sktime/sktime/issues)                  |
| **Usage Questions**                  | [GitHub Discussions](https://github.com/sktime/sktime/discussions) · [Stack Overflow](https://stackoverflow.com/questions/tagged/sktime) |
| **General Discussion**                | [GitHub Discussions](https://github.com/sktime/sktime/discussions)      |
| **Contribution & Development**        | `dev-chat` channel · [Discord](https://discord.com/invite/54ACzaFsn7)             |
| **Meet-ups and collaboration sessions** | [Discord](https://discord.com/invite/54ACzaFsn7) - Fridays 13 UTC, dev/meet-ups channel |

## sktime Features at a Glance

sktime offers a unified interface for various time series learning tasks, including:

*   **Forecasting:** Predict future values. [Tutorial](https://www.sktime.net/en/latest/examples/01_forecasting.html)
*   **Time Series Classification:** Categorize time series data. [Tutorial](https://github.com/sktime/sktime/blob/main/examples/02_classification.ipynb)
*   **Time Series Regression**
*   **Transformations**
*   **Detection Tasks**
*   **Parameter Fitting**
*   **Time Series Clustering**
*   **Time Series Distances/Kernels**
*   **Time Series Alignment**
*   **Time Series Splitters**
*   **Distributions and simulation**

## Installing sktime

For detailed installation instructions and troubleshooting, see the [documentation](https://www.sktime.net/en/latest/installation.html).

**Requirements:**

*   macOS X · Linux · Windows 8.1 or higher
*   Python 3.8, 3.9, 3.10, 3.11, and 3.12 (64-bit)

**Install using:**

*   **pip:** `pip install sktime` or `pip install sktime[all_extras]` for optional dependencies.
*   **conda:** `conda install -c conda-forge sktime` or `conda install -c conda-forge sktime-all-extras` for all dependencies.

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

Contribute to sktime and help build the best time series analysis toolkit!
Find out how you can get involved!

*   **[Contribute](https://www.sktime.net/en/latest/get_involved/contributing.html)**: Learn how to contribute.
*   **[Mentoring](https://github.com/sktime/mentoring)**: Join our mentoring program.
*   **[Meetings](https://calendar.google.com/calendar/u/0/embed?src=sktime.toolbox@gmail.com&ctz=UTC)**: Join our discussions, tutorials, workshops, and sprints!
*   **[Developer Guides](https://www.sktime.net/en/latest/developer_guide.html)**: Learn to develop sktime's codebase.
*   **[Enhancement Proposals](https://github.com/sktime/enhancement-proposals)**: Propose new features.
*   **[Contributors](https://github.com/sktime/sktime/blob/main/CONTRIBUTORS.md)**: See a list of all contributors.
*   **[Roles](https://www.sktime.net/en/latest/about/team.html)**: Learn about our community roles.
*   **[Donate](https://opencollective.com/sktime)**: Support sktime's development.
*   **[Governance](https://www.sktime.net/en/latest/get_involved/governance.html)**: Learn how decisions are made.

## sktime Hall of Fame

<a href="https://github.com/sktime/sktime/graphs/contributors">
<img src="https://opencollective.com/sktime/contributors.svg?width=600&button=false" />
</a>

## sktime's Vision

*   **Community-Driven:** Developed by a friendly and collaborative community.
*   **Task-Focused:** Provides the right tools for the right time series analysis tasks.
*   **Interoperable:** Integrates with scikit-learn, statsmodels, and other key libraries.
*   **Model Composition:** Supports building, tuning, and feature extraction pipelines.
*   **Clean Design:** Uses modern object-oriented design principles.
*   **Fair Assessment:** Emphasizes robust model assessment and benchmarking.
*   **Extensible:** Easy to extend with custom algorithms.

**[Visit the sktime repository on GitHub](https://github.com/sktime/sktime)**