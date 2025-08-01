# sktime: Your Comprehensive Toolkit for Time Series Analysis

[![sktime Logo](https://github.com/sktime/sktime/blob/main/docs/source/images/sktime-logo.svg?raw=true "sktime Logo")](https://www.sktime.net)

**sktime empowers you to master time series analysis with a unified and user-friendly Python library.**  

**[View the sktime Repository on GitHub](https://github.com/sktime/sktime)**

**Key Features:**

*   **Unified Interface:** Provides a consistent API for various time series learning tasks.
*   **Diverse Tasks:** Supports forecasting, classification, clustering, anomaly detection, and more.
*   **Extensive Algorithms:** Includes a wide range of time series algorithms.
*   **scikit-learn Compatibility:** Offers tools for building, tuning, and validating models, compatible with scikit-learn.
*   **Interoperability:** Interfaces with popular libraries like scikit-learn, statsmodels, tsfresh, and PyOD.
*   **Model Composition:** Enables building pipelines, ensembling, tuning, and reduction techniques.
*   **Easily Extensible:** Offers extension templates to add custom algorithms.

**What's New:**

*   **Latest Version:** **0.38.4**
*   **[Release Notes](https://www.sktime.net/en/latest/changelog.html)**

|  | **Links** |
|---|---|
| **Documentation** | [Documentation](https://www.sktime.net/en/stable/users.html) · [Tutorials](https://www.sktime.net/en/latest/tutorials.html) · [Examples](https://www.sktime.net/en/latest/examples.html) · [Release Notes](https://www.sktime.net/en/latest/changelog.html) |
| **Open Source** | [![BSD 3-clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/sktime/sktime/blob/main/LICENSE) [![GC.OS Sponsored](https://img.shields.io/badge/GC.OS-Sponsored%20Project-orange.svg?style=flat&colorA=0eac92&colorB=2077b4)](https://gc-os-ai.github.io/) |
| **Tutorials** | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sktime/sktime/main?filepath=examples) [![!youtube](https://img.shields.io/static/v1?logo=youtube&label=YouTube&message=tutorials&color=red)](https://www.youtube.com/playlist?list=PLKs3UgGjlWHqNzu0LEOeLKvnjvvest2d0) |
| **Community** | [![!discord](https://img.shields.io/static/v1?logo=discord&label=discord&message=chat&color=lightgreen)](https://discord.com/invite/54ACzaFsn7) [![!slack](https://img.shields.io/static/v1?logo=linkedin&label=LinkedIn&message=news&color=lightblue)](https://www.linkedin.com/company/scikit-time/)  |
| **CI/CD** | [![github-actions](https://img.shields.io/github/actions/workflow/status/sktime/sktime/wheels.yml?logo=github)](https://github.com/sktime/sktime/actions/workflows/wheels.yml) [![readthedocs](https://img.shields.io/readthedocs/sktime?logo=readthedocs)](https://www.sktime.net/en/latest/?badge=latest) [![platform](https://img.shields.io/conda/pn/conda-forge/sktime)](https://github.com/sktime/sktime) |
| **Code** |  [![!pypi](https://img.shields.io/pypi/v/sktime?color=orange)](https://pypi.org/project/sktime/) [![!conda](https://img.shields.io/conda/vn/conda-forge/sktime)](https://anaconda.org/conda-forge/sktime) [![!python-versions](https://img.shields.io/pypi/pyversions/sktime)](https://www.python.org/) [![!black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)  |
| **Downloads** | ![PyPI - Downloads](https://img.shields.io/pypi/dw/sktime) ![PyPI - Downloads](https://img.shields.io/pypi/dm/sktime) [![Downloads](https://static.pepy.tech/personalized-badge/sktime?period=total&units=international_system&left_color=grey&right_color=blue&left_text=cumulative%20(pypi))](https://pepy.tech/project/sktime) |
| **Citation** | [![!zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.3749000.svg)](https://doi.org/10.5281/zenodo.3749000) |

## 📚 Documentation

*   **[Tutorials](https://www.sktime.net/en/latest/tutorials.html):** Start your sktime journey.
*   **[Binder Notebooks](https://mybinder.org/v2/gh/sktime/sktime/main?filepath=examples):** Interactive examples in your browser.
*   **[Examples](https://www.sktime.net/en/latest/examples.html):** Practical usage of sktime features.
*   **[API Reference](https://www.sktime.net/en/latest/api_reference.html):** Detailed API reference.
*   **[Changelog](https://www.sktime.net/en/latest/changelog.html):** Version history and changes.
*   **[Roadmap](https://www.sktime.net/en/latest/roadmap.html):** Development plan.

## 💬 Where to Ask Questions

Get help and join the community:

*   **Bug Reports & Feature Requests:** [GitHub Issue Tracker](https://github.com/sktime/sktime/issues)
*   **Usage Questions & General Discussion:** [GitHub Discussions](https://github.com/sktime/sktime/discussions) · [Stack Overflow](https://stackoverflow.com/questions/tagged/sktime)
*   **Contribution & Development:** `dev-chat` channel · [Discord](https://discord.com/invite/54ACzaFsn7)
*   **Meet-ups and Collaboration:** [Discord](https://discord.com/invite/54ACzaFsn7) - Fridays 13 UTC, dev/meet-ups channel

## ✨ Features

sktime streamlines the time series analysis ecosystem by providing a __unified interface for various time series learning tasks__. It features [__dedicated time series algorithms__](https://www.sktime.net/en/stable/estimator_overview.html) and __tools for composite model building__,  such as pipelining, ensembling, tuning, and reduction, empowering users to apply algorithms designed for one task to another.

sktime also provides **interfaces to related libraries**, for example [scikit-learn], [statsmodels], [tsfresh], [PyOD], and [fbprophet], among others.

| Module | Status | Links |
|---|---|---|
| **[Forecasting]** | stable | [Tutorial](https://www.sktime.net/en/latest/examples/01_forecasting.html) · [API Reference](https://www.sktime.net/en/latest/api_reference/forecasting.html)  |
| **[Time Series Classification]** | stable | [Tutorial](https://github.com/sktime/sktime/blob/main/examples/02_classification.ipynb) · [API Reference](https://www.sktime.net/en/latest/api_reference/classification.html) |
| **[Time Series Regression]** | stable | [API Reference](https://www.sktime.net/en/latest/api_reference/regression.html) |
| **[Transformations]** | stable | [Tutorial](https://github.com/sktime/sktime/blob/main/examples/03_transformers.ipynb) · [API Reference](https://www.sktime.net/en/latest/api_reference/transformations.html) |
| **[Detection tasks]** | maturing |  |
| **[Parameter fitting]** | maturing | [API Reference](https://www.sktime.net/en/latest/api_reference/param_est.html)  |
| **[Time Series Clustering]** | maturing | [API Reference](https://www.sktime.net/en/latest/api_reference/clustering.html) |
| **[Time Series Distances/Kernels]** | maturing | [Tutorial](https://github.com/sktime/sktime/blob/main/examples/03_transformers.ipynb) · [API Reference](https://www.sktime.net/en/latest/api_reference/dists_kernels.html)  |
| **[Time Series Alignment]** | experimental | [API Reference](https://www.sktime.net/en/latest/api_reference/alignment.html) |
| **[Time Series Splitters]** | maturing |  |
| **[Distributions and simulation]** | experimental |  |

[forecasting]: https://github.com/sktime/sktime/tree/main/sktime/forecasting
[time series classification]: https://github.com/sktime/sktime/tree/main/sktime/classification
[time series regression]: https://github.com/sktime/sktime/tree/main/sktime/regression
[time series clustering]: https://github.com/sktime/sktime/tree/main/sktime/clustering
[detection tasks]: https://github.com/sktime/sktime/tree/main/sktime/detection
[time series distances/kernels]: https://github.com/sktime/sktime/tree/main/sktime/dists_kernels
[time series alignment]: https://github.com/sktime/sktime/tree/main/sktime/alignment
[transformations]: https://github.com/sktime/sktime/tree/main/sktime/transformations
[distributions and simulation]: https://github.com/sktime/sktime/tree/main/sktime/proba
[time series splitters]: https://github.com/sktime/sktime/tree/main/sktime/split
[parameter fitting]: https://github.com/sktime/sktime/tree/main/sktime/param_est

## 🚀 Installation

Follow these steps for a smooth installation:

*   **Operating System:** macOS X, Linux, Windows 8.1 or higher
*   **Python Version:** 3.8, 3.9, 3.10, 3.11, and 3.12 (64-bit only)
*   **Package Managers:** pip & conda (via `conda-forge`)

### pip

```bash
pip install sktime
```

For all extra dependencies, use:

```bash
pip install sktime[all_extras]
```

For specific task dependencies:

```bash
pip install sktime[forecasting]  # for selected forecasting dependencies
pip install sktime[forecasting,transformations]  # forecasters and transformers
```

Valid sets are:
* `forecasting`
* `transformations`
* `classification`
* `regression`
* `clustering`
* `param_est`
* `networks`
* `detection`
* `alignment`

### conda

```bash
conda install -c conda-forge sktime
```

For all extra dependencies, use:

```bash
conda install -c conda-forge sktime-all-extras
```

## 💡 Quickstart

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

## 🤝 Get Involved

Join the sktime community!  All contributions are welcome.

| **Get Involved** |
| -------------------------- |
| **[Contribute](https://www.sktime.net/en/latest/get_involved/contributing.html)** How to contribute. |
| **[Mentoring](https://github.com/sktime/mentoring)** Apply for mentoring. |
| **[Meetings](https://calendar.google.com/calendar/u/0/embed?src=sktime.toolbox@gmail.com&ctz=UTC)** Join discussions, tutorials, and sprints. |
| **[Developer Guides](https://www.sktime.net/en/latest/developer_guide.html)** Develop sktime's codebase. |
| **[Enhancement Proposals](https://github.com/sktime/enhancement-proposals)** Design a new feature. |
| **[Contributors](https://github.com/sktime/sktime/blob/main/CONTRIBUTORS.md)** List of contributors. |
| **[Roles](https://www.sktime.net/en/latest/about/team.html)** Core community roles. |
| **[Donate](https://opencollective.com/sktime)** Fund sktime development. |
| **[Governance](https://www.sktime.net/en/latest/get_involved/governance.html)** Community decision-making.  |

## 🎉 Hall of Fame

Thanks to our amazing community for your contributions!

<a href="https://github.com/sktime/sktime/graphs/contributors">
<img src="https://opencollective.com/sktime/contributors.svg?width=600&button=false" />
</a>

## 💡 Project Vision

*   **Community-Driven:** Developed by and for a friendly and collaborative community.
*   **Task-Oriented:** Provides the right tools for specific learning problems.
*   **Ecosystem Integration:** Interoperable with scikit-learn, statsmodels, tsfresh, and more.
*   **Rich Functionality:** Includes model composition, tuning, and feature extraction.
*   **Clear Design:** Modern, object-oriented design.
*   **Fair Assessment:** Model inspection and avoidance of pitfalls.
*   **Extensible:** Easy-to-use templates for adding your own algorithms.