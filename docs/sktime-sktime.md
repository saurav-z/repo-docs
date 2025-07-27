# sktime: The Unified Toolkit for Time Series Analysis

[![sktime Logo](https://github.com/sktime/sktime/blob/main/docs/source/images/sktime-logo.svg?raw=true "sktime logo")](https://www.sktime.net)

**sktime is your one-stop shop for time series analysis in Python, offering a unified interface for various time series learning tasks!**  ([Explore the sktime repository](https://github.com/sktime/sktime))

*Version 0.38.4* is out now!  [Read the Release Notes](https://www.sktime.net/en/latest/changelog.html)

## Key Features

*   **Unified Interface:** Standardized API for time series forecasting, classification, clustering, anomaly detection, and more.
*   **Comprehensive Algorithms:** Built-in algorithms for diverse time series tasks.
*   **scikit-learn Compatibility:** Seamless integration with scikit-learn tools for model building, tuning, and validation.
*   **Modular Design:** Easily build and customize time series models using pipelines and ensembles.
*   **Extensible:**  Templates available to add your own algorithms compatible with sktime's API.

## Core Functionality & Tasks

sktime provides tools for the following time series tasks:

*   **Forecasting:** Predict future values of time series data.
*   **Time Series Classification:** Categorize time series into predefined classes.
*   **Time Series Regression:** Predict continuous values from time series data.
*   **Anomaly/Changepoint Detection:** Identify unusual patterns or shifts in time series data.
*   **Clustering:** Group similar time series together.
*   **Transformation:** Apply various transformations to time series data.

## Getting Started

### Documentation

| Section                    | Description                                       |
| -------------------------- | ------------------------------------------------- |
| **[Tutorials](https://www.sktime.net/en/latest/tutorials.html)**               | Start here to learn the basics!                  |
| **[Examples](https://www.sktime.net/en/latest/examples.html)**                  | Practical code examples.                         |
| **[API Reference](https://www.sktime.net/en/latest/api_reference.html)**        | Detailed API documentation.                     |
| **[Changelog](https://www.sktime.net/en/latest/changelog.html)**               | See what's new in each version.                |
| **[Roadmap](https://www.sktime.net/en/latest/roadmap.html)**                 | Future development plans.                           |
| **[Related Software](https://www.sktime.net/en/latest/related_software.html)**      | List of related software. |

### Installation

*   **Supported Operating Systems:** macOS, Linux, Windows (8.1 or higher).
*   **Python Version:** 3.8, 3.9, 3.10, 3.11, and 3.12 (64-bit).
*   **Package Managers:** pip and conda (via conda-forge).

#### pip

```bash
pip install sktime
```

For maximum dependencies:

```bash
pip install sktime[all_extras]
```

For dependencies for specific tasks:

```bash
pip install sktime[forecasting]
pip install sktime[forecasting,transformations]
```

#### conda

```bash
conda install -c conda-forge sktime
```

To install with all extras, use:

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

## Modules

| Module                       | Status   | Links                                                                                               |
| ---------------------------- | -------- | --------------------------------------------------------------------------------------------------- |
| **Forecasting**              | stable   | [Tutorial](https://www.sktime.net/en/latest/examples/01_forecasting.html) · [API Reference](https://www.sktime.net/en/latest/api_reference/forecasting.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/forecasting.py) |
| **Time Series Classification** | stable   | [Tutorial](https://github.com/sktime/sktime/blob/main/examples/02_classification.ipynb) · [API Reference](https://www.sktime.net/en/latest/api_reference/classification.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/classification.py) |
| **Time Series Regression**   | stable   | [API Reference](https://www.sktime.net/en/latest/api_reference/regression.html)                |
| **Transformations**          | stable   | [Tutorial](https://github.com/sktime/sktime/blob/main/examples/03_transformers.ipynb) · [API Reference](https://www.sktime.net/en/latest/api_reference/transformations.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/transformer.py) |
| **Detection Tasks**          | maturing | [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/detection.py)           |
| **Parameter Fitting**        | maturing | [API Reference](https://www.sktime.net/en/latest/api_reference/param_est.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/transformer.py) |
| **Time Series Clustering**   | maturing | [API Reference](https://www.sktime.net/en/latest/api_reference/clustering.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/clustering.py) |
| **Distances/Kernels**        | maturing | [Tutorial](https://github.com/sktime/sktime/blob/main/examples/03_transformers.ipynb) · [API Reference](https://www.sktime.net/en/latest/api_reference/dists_kernels.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/dist_kern_panel.py) |
| **Time Series Alignment**    | experimental | [API Reference](https://www.sktime.net/en/latest/api_reference/alignment.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/alignment.py) |
| **Splitters**                | maturing | [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/split.py)           |
| **Distributions/Simulation** | experimental |                                                                                                   |

## Community & Getting Involved

Join the vibrant sktime community!  We welcome contributions of all kinds.

### Resources

| Resource                      | Description                                        |
| ----------------------------- | -------------------------------------------------- |
| **[Contribute](https://www.sktime.net/en/latest/get_involved/contributing.html)**      | How to contribute to sktime.           |
| **[Mentoring](https://github.com/sktime/mentoring)**         | Apply to our mentoring program!        |
| **[Meetings](https://calendar.google.com/calendar/u/0/embed?src=sktime.toolbox@gmail.com&ctz=UTC)**   | Join discussions, tutorials, and sprints! |
| **[Developer Guides](https://www.sktime.net/en/latest/developer_guide.html)**           | Develop sktime's codebase.                |
| **[Enhancement Proposals](https://github.com/sktime/enhancement-proposals)**         | Design a new feature.                   |
| **[Contributors](https://github.com/sktime/sktime/blob/main/CONTRIBUTORS.md)**        | List of all contributors.               |
| **[Roles](https://www.sktime.net/en/latest/about/team.html)**             | Community roles.                       |
| **[Donate](https://opencollective.com/sktime)**                      | Support sktime's development.          |
| **[Governance](https://www.sktime.net/en/latest/get_involved/governance.html)**        | How decisions are made.                  |

### Communication

*   **Bug Reports & Feature Requests:** [GitHub Issue Tracker](https://github.com/sktime/sktime/issues)
*   **Usage Questions & General Discussion:** [GitHub Discussions](https://github.com/sktime/sktime/discussions) · [Stack Overflow](https://stackoverflow.com/questions/tagged/sktime)
*   **Contribution & Development:** dev-chat channel · [Discord](https://discord.com/invite/54ACzaFsn7)
*   **Meet-ups:** [Discord](https://discord.com/invite/54ACzaFsn7) - Fridays 13 UTC, dev/meet-ups channel

## Project Vision

*   **Community-Driven:** Developed by and for a collaborative community.
*   **Task-Focused:** Provides the right tools for the right time series tasks.
*   **Interoperable:** Integrates with leading data science ecosystems.
*   **Modular & Extensible:** Supports building complex models and incorporating custom algorithms.
*   **Robust:**  Provides fair model assessment and benchmarking.

## Hall of Fame

[Open Collective Contributors](https://opencollective.com/sktime/contributors.svg?width=600&button=false)

**Thank you to all our contributors!**