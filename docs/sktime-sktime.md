# sktime: Your Unified Toolkit for Time Series Analysis

<a href="https://www.sktime.net"><img src="https://github.com/sktime/sktime/blob/main/docs/source/images/sktime-logo.svg" width="175" align="right" /></a>

**Tackle time series challenges with ease using sktime, a Python library offering a unified interface for diverse time series learning tasks.** ( [Visit the sktime Repository](https://github.com/sktime/sktime) )

*   **Version 0.38.4 is now available!** [Check out the release notes here](https://www.sktime.net/en/latest/changelog.html).

**Key Features:**

*   **Unified Interface:** Provides a consistent API for forecasting, classification, clustering, anomaly detection, and more.
*   **Comprehensive Algorithms:** Includes a wide range of time series algorithms.
*   **scikit-learn Compatibility:** Integrates seamlessly with scikit-learn tools for model building, tuning, and validation.
*   **Extensive Documentation:** Offers tutorials, examples, and API references.
*   **Community Driven:** Developed and maintained by a collaborative community.

## Table of Contents

*   [Documentation](#documentation)
*   [Features](#features)
*   [Install sktime](#install-sktime)
*   [Quickstart](#quickstart)
*   [How to get involved](#how-to-get-involved)
*   [Project Vision](#project-vision)

## Documentation

Explore comprehensive resources to get started with sktime:

*   **:star: [Tutorials](https://www.sktime.net/en/latest/tutorials.html)**: Step-by-step guides for new users.
*   **:clipboard: [Binder Notebooks](https://mybinder.org/v2/gh/sktime/sktime/main?filepath=examples)**: Interactive examples to try in your browser.
*   **:woman_technologist: [Examples](https://www.sktime.net/en/latest/examples.html)**: Practical demonstrations of sktime features.
*   **:scissors: [Extension Templates](https://github.com/sktime/sktime/blob/main/extension_templates)**: Create custom estimators with sktime's API.
*   **:control_knobs: [API Reference](https://www.sktime.net/en/latest/api_reference.html)**: Detailed documentation of the sktime API.
*   **:tv: [Video Tutorial](https://github.com/sktime/sktime-tutorial-pydata-global-2021)**: A comprehensive video tutorial.
*   **:hammer_and_wrench: [Changelog](https://www.sktime.net/en/latest/changelog.html)**: View the release history and changes.
*   **:deciduous_tree: [Roadmap](https://www.sktime.net/en/latest/roadmap.html)**: sktime's plan for development.
*   **:pencil: [Related Software](https://www.sktime.net/en/latest/related_software.html)**: Additional useful software.

## Features

sktime offers a wide range of tools for time series analysis, including:

*   Forecasting
*   Time Series Classification
*   Time Series Regression
*   Transformations
*   Anomaly/Change Point Detection
*   Parameter Fitting
*   Time Series Clustering
*   Time Series Distances/Kernels
*   Time Series Alignment
*   Time Series Splitters
*   Distributions and Simulation

## Install sktime

Get started with sktime:

*   **Operating Systems:** macOS X, Linux, and Windows 8.1 or higher
*   **Python Version:** Python 3.8, 3.9, 3.10, 3.11, and 3.12 (64-bit only)
*   **Package Managers:** pip and conda (via conda-forge)

### pip

```bash
pip install sktime
```

or, install all dependencies:

```bash
pip install sktime[all_extras]
```

For specific learning tasks:

```bash
pip install sktime[forecasting]  # for selected forecasting dependencies
pip install sktime[forecasting,transformations]  # forecasters and transformers
```

Valid sets:
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

or, install all dependencies:

```bash
conda install -c conda-forge sktime-all-extras
```

## Quickstart

**Forecasting:**

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

**Time Series Classification:**

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

## How to get involved

Join the sktime community and contribute:

*   **:gift_heart: [Contribute](https://www.sktime.net/en/latest/get_involved/contributing.html)**: Learn how to contribute.
*   **:school_satchel:  [Mentoring](https://github.com/sktime/mentoring)**: Join the mentoring program.
*   **:date: [Meetings](https://calendar.google.com/calendar/u/0/embed?src=sktime.toolbox@gmail.com&ctz=UTC)**: Participate in discussions and events.
*   **:woman_mechanic:  [Developer Guides](https://www.sktime.net/en/latest/developer_guide.html)**: Learn how to develop the codebase.
*   **:construction: [Enhancement Proposals](https://github.com/sktime/enhancement-proposals)**: Propose new features.
*   **:medal_sports: [Contributors](https://github.com/sktime/sktime/blob/main/CONTRIBUTORS.md)**: See a list of contributors.
*   **:raising_hand: [Roles](https://www.sktime.net/en/latest/about/team.html)**: Learn about community roles.
*   **:money_with_wings: [Donate](https://opencollective.com/sktime)**: Support sktime's development.
*   **:classical_building: [Governance](https://www.sktime.net/en/latest/get_involved/governance.html)**: Learn how decisions are made.

## Project Vision

*   **Community-Driven:** Developed by a friendly and collaborative community.
*   **Task-Specific Solutions:** Guides users to the right models for their problems.
*   **Interoperable:** Works with scikit-learn, statsmodels, tsfresh, and more.
*   **Model Composition:** Provides rich functionality for building and tuning pipelines.
*   **Clean Syntax:** Based on modern object-oriented design.
*   **Fair Assessment:** Enables model inspection and avoidance of pitfalls.
*   **Extensible:** Offers easy extension templates for custom algorithms.