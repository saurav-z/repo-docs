# sktime: The Unified Toolkit for Time Series Analysis

[![sktime Logo](https://github.com/sktime/sktime/blob/main/docs/source/images/sktime-logo.svg?raw=true)](https://github.com/sktime/sktime)

**sktime empowers you to easily build, evaluate, and deploy time series models for forecasting, classification, and more.** ([Original Repository](https://github.com/sktime/sktime))

*   🚀 **Version 0.38.4 is now available!** [See the Release Notes](https://www.sktime.net/en/latest/changelog.html).

sktime is a comprehensive Python library designed for all things time series analysis. It offers a unified interface for a wide array of time series learning tasks, including forecasting, classification, clustering, anomaly/changepoint detection, and more. Built with compatibility with [scikit-learn](https://scikit-learn.org/stable/), sktime provides a robust framework for building, tuning, and validating time series models with both pre-built [time series algorithms](https://www.sktime.net/en/stable/estimator_overview.html) and custom tools.

## Key Features

*   **Unified Interface:** Simplify your time series analysis workflow with a consistent API for diverse tasks.
*   **Comprehensive Task Support:** Tackle forecasting, classification, regression, clustering, anomaly detection, and more.
*   **Extensive Algorithm Library:** Access a wide range of pre-built time series algorithms, including those compatible with scikit-learn.
*   **Model Building Tools:** Utilize powerful tools for model composition, ensembling, hyperparameter tuning, and feature engineering.
*   **Integration with Popular Libraries:** Seamlessly integrate with scikit-learn, statsmodels, tsfresh, and other essential tools.
*   **Easy Extensibility:** Extend sktime's functionality by adding custom algorithms using provided templates.

## Documentation & Resources

Explore the comprehensive resources available to get you started with sktime:

*   :star: **[Tutorials](https://www.sktime.net/en/latest/tutorials.html)**: New to sktime? Here's everything you need to know!
*   :clipboard: **[Binder Notebooks](https://mybinder.org/v2/gh/sktime/sktime/main?filepath=examples)**: Example notebooks to play with in your browser.
*   :woman_technologist: **[Examples](https://www.sktime.net/en/latest/examples.html)**: How to use sktime and its features.
*   :scissors: **[Extension Templates](https://github.com/sktime/sktime/blob/main/extension_templates)**: How to build your own estimator using sktime's API.
*   :control_knobs: **[API Reference](https://www.sktime.net/en/latest/api_reference.html)**: The detailed reference for sktime's API.
*   :tv: **[Video Tutorial](https://github.com/sktime/sktime-tutorial-pydata-global-2021)**: Our video tutorial from 2021 PyData Global.
*   :hammer_and_wrench: **[Changelog](https://www.sktime.net/en/latest/changelog.html)**: Changes and version history.
*   :deciduous_tree: **[Roadmap](https://www.sktime.net/en/latest/roadmap.html)**: sktime's software and community development plan.
*   :pencil: **[Related Software](https://www.sktime.net/en/latest/related_software.html)**: A list of related software.

## Get Involved

Join the thriving sktime community and contribute to the project:

*   :gift_heart: **[Contribute](https://www.sktime.net/en/latest/get_involved/contributing.html)**: How to contribute to sktime.
*   :school_satchel: **[Mentoring](https://github.com/sktime/mentoring)**: New to open source? Apply to our mentoring program!
*   :date: **[Meetings](https://calendar.google.com/calendar/u/0/embed?src=sktime.toolbox@gmail.com&ctz=UTC)**: Join our discussions, tutorials, workshops, and sprints!
*   :woman_mechanic: **[Developer Guides](https://www.sktime.net/en/latest/developer_guide.html)**: How to further develop sktime's code base.
*   :construction: **[Enhancement Proposals](https://github.com/sktime/enhancement-proposals)**: Design a new feature for sktime.
*   :medal_sports: **[Contributors](https://github.com/sktime/sktime/blob/main/CONTRIBUTORS.md)**: A list of all contributors.
*   :raising_hand: **[Roles](https://www.sktime.net/en/latest/about/team.html)**: An overview of our core community roles.
*   :money_with_wings: **[Donate](https://opencollective.com/sktime)**: Fund sktime maintenance and development.
*   :classical_building: **[Governance](https://www.sktime.net/en/latest/get_involved/governance.html)**: How and by whom decisions are made in sktime's community.

## Where to Get Help

Get your questions answered and connect with the community on the following platforms:

*   :bug: **Bug Reports:** [GitHub Issue Tracker](https://github.com/sktime/sktime/issues)
*   :sparkles: **Feature Requests & Ideas:** [GitHub Issue Tracker](https://github.com/sktime/sktime/issues)
*   :woman_technologist: **Usage Questions:** [GitHub Discussions](https://github.com/sktime/sktime/discussions) · [Stack Overflow](https://stackoverflow.com/questions/tagged/sktime)
*   :speech_balloon: **General Discussion:** [GitHub Discussions](https://github.com/sktime/sktime/discussions)
*   :factory: **Contribution & Development:** `dev-chat` channel · [Discord](https://discord.com/invite/54ACzaFsn7)
*   :globe_with_meridians: **Meet-ups and collaboration sessions:** [Discord](https://discord.com/invite/54ACzaFsn7) - Fridays 13 UTC, dev/meet-ups channel

## Code & Package Details

*   **PyPI:** [![PyPI version](https://img.shields.io/pypi/v/sktime?color=orange)](https://pypi.org/project/sktime/)
*   **Conda-forge:** [![Conda Version](https://img.shields.io/conda/vn/conda-forge/sktime)](https://anaconda.org/conda-forge/sktime)
*   **Python Versions:** [![Python Versions](https://img.shields.io/pypi/pyversions/sktime)](https://www.python.org/)
*   **Code Style:** [![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
*   **Downloads:**
    [![PyPI Downloads](https://img.shields.io/pypi/dw/sktime)](https://pypi.org/project/sktime/)
    [![PyPI Downloads](https://img.shields.io/pypi/dm/sktime)](https://pypi.org/project/sktime/)
    [![Cumulative Downloads](https://static.pepy.tech/personalized-badge/sktime?period=total&units=international_system&left_color=grey&right_color=blue&left_text=cumulative%20(pypi))](https://pepy.tech/project/sktime)
*   **Citation:** [![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3749000.svg)](https://doi.org/10.5281/zenodo.3749000)

## Installation

*   **Operating Systems:** macOS X · Linux · Windows 8.1 or higher
*   **Python:** 3.8, 3.9, 3.10, 3.11, and 3.12 (only 64-bit)
*   **Package Managers:** [pip](https://pip.pypa.io/en/stable/) · [conda](https://docs.conda.io/en/latest/) (via `conda-forge`)

### pip

```bash
pip install sktime
```

Install with all extra dependencies:

```bash
pip install sktime[all_extras]
```

or for curated sets of soft dependencies for specific learning tasks:

```bash
pip install sktime[forecasting]  # for selected forecasting dependencies
pip install sktime[forecasting,transformations]  # forecasters and transformers
```

Valid sets are: `forecasting`, `transformations`, `classification`, `regression`, `clustering`, `param_est`, `networks`, `detection`, `alignment`.

### conda

```bash
conda install -c conda-forge sktime
```

Install with all extra dependencies:

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

## Modules & Functionality

| Module                           | Status     | Links                                                                                                |
| :------------------------------- | :--------- | :--------------------------------------------------------------------------------------------------- |
| **[Forecasting](https://github.com/sktime/sktime/tree/main/sktime/forecasting)**           | stable     | [Tutorial](https://www.sktime.net/en/latest/examples/01_forecasting.html) · [API Reference](https://www.sktime.net/en/latest/api_reference/forecasting.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/forecasting.py)  |
| **[Time Series Classification](https://github.com/sktime/sktime/tree/main/sktime/classification)** | stable     | [Tutorial](https://github.com/sktime/sktime/blob/main/examples/02_classification.ipynb) · [API Reference](https://www.sktime.net/en/latest/api_reference/classification.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/classification.py) |
| **[Time Series Regression](https://github.com/sktime/sktime/tree/main/sktime/regression)**  | stable     | [API Reference](https://www.sktime.net/en/latest/api_reference/regression.html)                                                             |
| **[Transformations](https://github.com/sktime/sktime/tree/main/sktime/transformations)**   | stable     | [Tutorial](https://github.com/sktime/sktime/blob/main/examples/03_transformers.ipynb) · [API Reference](https://www.sktime.net/en/latest/api_reference/transformations.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/transformer.py)  |
| **[Detection Tasks](https://github.com/sktime/sktime/tree/main/sktime/detection)**           | maturing   | [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/detection.py)                                           |
| **[Parameter Fitting](https://github.com/sktime/sktime/tree/main/sktime/param_est)**     | maturing   | [API Reference](https://www.sktime.net/en/latest/api_reference/param_est.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/transformer.py)  |
| **[Time Series Clustering](https://github.com/sktime/sktime/tree/main/sktime/clustering)**  | maturing   | [API Reference](https://www.sktime.net/en/latest/api_reference/clustering.html) ·  [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/clustering.py) |
| **[Time Series Distances/Kernels](https://github.com/sktime/sktime/tree/main/sktime/dists_kernels)** | maturing   | [Tutorial](https://github.com/sktime/sktime/blob/main/examples/03_transformers.ipynb) · [API Reference](https://www.sktime.net/en/latest/api_reference/dists_kernels.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/dist_kern_panel.py) |
| **[Time Series Alignment](https://github.com/sktime/sktime/tree/main/sktime/alignment)**    | experimental | [API Reference](https://www.sktime.net/en/latest/api_reference/alignment.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/alignment.py) |
| **[Time Series Splitters](https://github.com/sktime/sktime/tree/main/sktime/split)**       | maturing   | [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/split.py)                                               |
| **[Distributions and simulation](https://github.com/sktime/sktime/tree/main/sktime/proba)** | experimental |                                                                                                                                 |

## Contributors

A huge thank you to all of our contributors!

<a href="https://github.com/sktime/sktime/graphs/contributors">
<img src="https://opencollective.com/sktime/contributors.svg?width=600&button=false" />
</a>

## Project Vision

*   **Community-Driven:** Developed by a welcoming and collaborative community.
*   **Task-Specific Tools:** Guides users to choose the right tools for their specific time series tasks.
*   **Interoperable:** Integrates seamlessly with scikit-learn, statsmodels, tsfresh, and other key libraries.
*   **Rich Composition:** Provides comprehensive model composition and reduction capabilities.
*   **Clean Syntax:** Employs a modern, object-oriented design for easy use.
*   **Fair Evaluation:** Provides robust tools for fair model assessment and benchmarking.
*   **Extensible:** Offers easy extension templates to add custom algorithms.