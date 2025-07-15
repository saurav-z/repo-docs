<a href="https://www.sktime.net"><img src="https://github.com/sktime/sktime/blob/main/docs/source/images/sktime-logo.svg" width="175" align="right" alt="sktime logo"/></a>

# sktime: The Unified Toolkit for Time Series Analysis

**Tackle diverse time series analysis tasks with ease using sktime, a Python library providing a unified interface for machine learning.**

[Explore the sktime repository on GitHub](https://github.com/sktime/sktime)

*   **Key Features**:
    *   **Unified Interface:** Simplifies time series analysis workflows with a consistent API.
    *   **Diverse Tasks:** Supports forecasting, classification, clustering, anomaly detection, and more.
    *   **Rich Algorithms:** Includes a wide range of time series algorithms.
    *   **scikit-learn Compatibility:** Integrates seamlessly with scikit-learn tools for model building, tuning, and validation.
    *   **Extensible:**  Easy to extend with your own algorithms.
    *   **Interoperable:** Interfaces with popular libraries like scikit-learn, statsmodels, tsfresh, and PyOD.

*   **[Documentation](https://www.sktime.net/en/stable/users.html)** · **[Tutorials](https://www.sktime.net/en/stable/examples.html)** · **[Release Notes](https://www.sktime.net/en/latest/changelog.html)**

| **Resources**                                     |                                                                                                                                             |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| **Open Source**                                   | [![BSD 3-clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/sktime/sktime/blob/main/LICENSE)  [![GC.OS Sponsored](https://img.shields.io/badge/GC.OS-Sponsored%20Project-orange.svg?style=flat&colorA=0eac92&colorB=2077b4)](https://gc-os-ai.github.io/) |
| **Tutorials & Community**                         | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sktime/sktime/main?filepath=examples) [![!youtube](https://img.shields.io/static/v1?logo=youtube&label=YouTube&message=tutorials&color=red)](https://www.youtube.com/playlist?list=PLKs3UgGjlWHqNzu0LEOeLKvnjvvest2d0) [![!discord](https://img.shields.io/static/v1?logo=discord&label=discord&message=chat&color=lightgreen)](https://discord.com/invite/54ACzaFsn7) [![!slack](https://img.shields.io/static/v1?logo=linkedin&label=LinkedIn&message=news&color=lightblue)](https://www.linkedin.com/company/scikit-time/)  |
| **CI/CD & Package Information**                    | [![github-actions](https://img.shields.io/github/actions/workflow/status/sktime/sktime/wheels.yml?logo=github)](https://github.com/sktime/sktime/actions/workflows/wheels.yml) [![readthedocs](https://img.shields.io/readthedocs/sktime?logo=readthedocs)](https://www.sktime.net/en/latest/?badge=latest) [![platform](https://img.shields.io/conda/pn/conda-forge/sktime)](https://github.com/sktime/sktime) [![!pypi](https://img.shields.io/pypi/v/sktime?color=orange)](https://pypi.org/project/sktime/) [![!conda](https://img.shields.io/conda/vn/conda-forge/sktime)](https://anaconda.org/conda-forge/sktime) [![!python-versions](https://img.shields.io/pypi/pyversions/sktime)](https://www.python.org/) [![!black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)  |
| **Downloads & Citation**                         | ![PyPI - Downloads](https://img.shields.io/pypi/dw/sktime) ![PyPI - Downloads](https://img.shields.io/pypi/dm/sktime) [![Downloads](https://static.pepy.tech/personalized-badge/sktime?period=total&units=international_system&left_color=grey&right_color=blue&left_text=cumulative%20(pypi))](https://pepy.tech/project/sktime) [![!zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.3749000.svg)](https://doi.org/10.5281/zenodo.3749000) |

## Documentation

Comprehensive documentation is available to help you get started:

*   **[Tutorials](https://www.sktime.net/en/latest/tutorials.html):**  Learn the fundamentals of sktime.
*   **[Binder Notebooks](https://mybinder.org/v2/gh/sktime/sktime/main?filepath=examples):** Interactive examples to experiment with in your browser.
*   **[Examples](https://www.sktime.net/en/latest/examples.html):**  Practical examples of how to use sktime features.
*   **[API Reference](https://www.sktime.net/en/latest/api_reference.html):**  Detailed information on the sktime API.
*   **[Changelog](https://www.sktime.net/en/latest/changelog.html):**  View the changes and version history.
*   **[Roadmap](https://www.sktime.net/en/latest/roadmap.html):**  Explore sktime's development plans.
*   **[Related Software](https://www.sktime.net/en/latest/related_software.html):** Discover related software.

## Community and Support

Get help and connect with the sktime community:

| Type                            | Platforms                               |
| ------------------------------- | --------------------------------------- |
| :bug: **Bug Reports**              | [GitHub Issue Tracker]                  |
| :sparkles: **Feature Requests & Ideas** | [GitHub Issue Tracker]                       |
| :woman_technologist: **Usage Questions**          | [GitHub Discussions] · [Stack Overflow] |
| :speech_balloon: **General Discussion**        | [GitHub Discussions] |
| :factory: **Contribution & Development** | `dev-chat` channel · [Discord] |
| :globe_with_meridians: **Meet-ups and collaboration sessions** | [Discord] - Fridays 13 UTC, dev/meet-ups channel |

[github issue tracker]: https://github.com/sktime/sktime/issues
[github discussions]: https://github.com/sktime/sktime/discussions
[stack overflow]: https://stackoverflow.com/questions/tagged/sktime
[discord]: https://discord.com/invite/54ACzaFsn7

## Features

sktime provides a unified interface for various time series learning tasks, including forecasting, classification, and anomaly detection.  It includes dedicated algorithms and tools for model building, tuning, and evaluation.  sktime also provides interfaces to related libraries.

### Core Functionality by Module

| Module                              | Status   | Links                                                                                                                 |
| ----------------------------------- | -------- | --------------------------------------------------------------------------------------------------------------------- |
| **Forecasting**                     | stable   | [Tutorial](https://www.sktime.net/en/latest/examples/01_forecasting.html) · [API Reference](https://www.sktime.net/en/latest/api_reference/forecasting.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/forecasting.py)  |
| **Time Series Classification**        | stable   | [Tutorial](https://github.com/sktime/sktime/blob/main/examples/02_classification.ipynb) · [API Reference](https://www.sktime.net/en/latest/api_reference/classification.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/classification.py) |
| **Time Series Regression**          | stable   | [API Reference](https://www.sktime.net/en/latest/api_reference/regression.html)                                        |
| **Transformations**                   | stable   | [Tutorial](https://github.com/sktime/sktime/blob/main/examples/03_transformers.ipynb) · [API Reference](https://www.sktime.net/en/latest/api_reference/transformations.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/transformer.py)  |
| **Detection tasks**                | maturing | [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/detection.py)                    |
| **Parameter fitting**                | maturing | [API Reference](https://www.sktime.net/en/latest/api_reference/param_est.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/transformer.py)  |
| **Time Series Clustering**          | maturing | [API Reference](https://www.sktime.net/en/latest/api_reference/clustering.html) ·  [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/clustering.py) |
| **Time Series Distances/Kernels**   | maturing | [Tutorial](https://github.com/sktime/sktime/blob/main/examples/03_transformers.ipynb) · [API Reference](https://www.sktime.net/en/latest/api_reference/dists_kernels.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/dist_kern_panel.py) |
| **Time Series Alignment**          | experimental | [API Reference](https://www.sktime.net/en/latest/api_reference/alignment.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/alignment.py) |
| **Time Series Splitters**          | maturing | [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/split.py) | |
| **Distributions and simulation** | experimental |  |

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

## Installation

Follow these steps to install sktime:

*   **Operating System:** macOS X, Linux, or Windows 8.1 or higher
*   **Python Version:** Python 3.8, 3.9, 3.10, 3.11, and 3.12 (64-bit)
*   **Package Managers:** [pip] or [conda] (via `conda-forge`)

[pip]: https://pip.pypa.io/en/stable/
[conda]: https://docs.conda.io/en/latest/

### Install with pip

```bash
pip install sktime
```

Install with all extras:

```bash
pip install sktime[all_extras]
```

Install with curated sets of soft dependencies for specific learning tasks:

```bash
pip install sktime[forecasting]  # for selected forecasting dependencies
pip install sktime[forecasting,transformations]  # forecasters and transformers
```

### Install with conda

```bash
conda install -c conda-forge sktime
```

Install with conda, with maximum dependencies:

```bash
conda install -c conda-forge sktime-all-extras
```

## Quickstart

Here are examples of how to use sktime for common tasks:

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

## Get Involved

Join the sktime community and contribute!

| Documentation              |                                                                |
| -------------------------- | --------------------------------------------------------------        |
| :gift_heart: **[Contribute]**        | How to contribute to sktime.          |
| :school_satchel:  **[Mentoring]** | New to open source? Apply to our mentoring program! |
| :date: **[Meetings]** | Join our discussions, tutorials, workshops, and sprints! |
| :woman_mechanic:  **[Developer Guides]**      | How to further develop sktime's code base.                             |
| :construction: **[Enhancement Proposals]** | Design a new feature for sktime. |
| :medal_sports: **[Contributors]** | A list of all contributors. |
| :raising_hand: **[Roles]** | An overview of our core community roles. |
| :money_with_wings: **[Donate]** | Fund sktime maintenance and development. |
| :classical_building: **[Governance]** | How and by whom decisions are made in sktime's community.   |

[contribute]: https://www.sktime.net/en/latest/get_involved/contributing.html
[donate]: https://opencollective.com/sktime
[extension templates]: https://github.com/sktime/sktime/tree/main/extension_templates
[developer guides]: https://www.sktime.net/en/latest/developer_guide.html
[contributors]: https://github.com/sktime/sktime/blob/main/CONTRIBUTORS.md
[governance]: https://www.sktime.net/en/latest/get_involved/governance.html
[mentoring]: https://github.com/sktime/mentoring
[meetings]: https://calendar.google.com/calendar/u/0/embed?src=sktime.toolbox@gmail.com&ctz=UTC
[enhancement proposals]: https://github.com/sktime/enhancement-proposals
[roles]: https://www.sktime.net/en/latest/about/team.html

## Hall of Fame

A special thank you to all our community members!

<a href="https://github.com/sktime/sktime/graphs/contributors">
<img src="https://opencollective.com/sktime/contributors.svg?width=600&button=false" />
</a>
<br>

## Project Vision

*   **Community-Driven:** Developed by a friendly and collaborative community.
*   **Task-Oriented:**  Helps users select the right models for their time series tasks.
*   **Interoperable:** Compatible with scikit-learn, statsmodels, tsfresh, and other libraries.
*   **Comprehensive:** Offers rich model composition and evaluation capabilities.
*   **Clean and Extensible:** Based on modern design principles for easy use and expansion.