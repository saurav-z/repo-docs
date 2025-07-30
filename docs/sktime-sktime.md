# sktime: A Unified Toolkit for Time Series Machine Learning

[![sktime Logo](https://github.com/sktime/sktime/blob/main/docs/source/images/sktime-logo.svg?raw=true)](https://www.sktime.net)

**Tackle your time series challenges with sktime, a Python library offering a unified interface for time series analysis, forecasting, and more!**  ([View on GitHub](https://github.com/sktime/sktime))

*   **Version 0.38.4 is now available!** [Check out the release notes](https://www.sktime.net/en/latest/changelog.html).

## Key Features

*   **Unified Interface:** Provides a consistent API for various time series tasks, simplifying your workflow.
*   **Comprehensive Tasks:** Supports forecasting, classification, clustering, anomaly detection, and more.
*   **Rich Algorithm Selection:** Includes a wide range of time series algorithms.
*   **Scikit-learn Compatibility:** Integrates seamlessly with scikit-learn tools for model building, tuning, and validation.
*   **Model Composition:** Offers tools for building composite models, including pipelining, ensembling, and tuning.
*   **Interoperability:** Interfaces with popular libraries like scikit-learn, statsmodels, and tsfresh.
*   **Extensible:** Easy to extend with your own algorithms using provided templates.

## Documentation

*   **[Tutorials](https://www.sktime.net/en/latest/tutorials.html):** Get started with sktime.
*   **[Examples](https://www.sktime.net/en/latest/examples.html):** Learn how to use sktime's features.
*   **[API Reference](https://www.sktime.net/en/latest/api_reference.html):** Detailed API documentation.
*   **[Changelog](https://www.sktime.net/en/latest/changelog.html):** View changes and version history.
*   **[Roadmap](https://www.sktime.net/en/latest/roadmap.html):** sktime's development plan.

## Key Resources

*   **Open Source:** [BSD 3-clause License](https://github.com/sktime/sktime/blob/main/LICENSE)
*   **Community:** [Discord Chat](https://discord.com/invite/54ACzaFsn7) | [LinkedIn](https://www.linkedin.com/company/scikit-time/)
*   **Tutorials:** [Binder](https://mybinder.org/v2/gh/sktime/sktime/main?filepath=examples) | [YouTube](https://www.youtube.com/playlist?list=PLKs3UgGjlWHqNzu0LEOeLKvnjvvest2d0)

## Support & Community

*   **Bug Reports & Feature Requests:** [GitHub Issue Tracker](https://github.com/sktime/sktime/issues)
*   **Discussions:** [GitHub Discussions](https://github.com/sktime/sktime/discussions)
*   **Usage Questions:** [Stack Overflow](https://stackoverflow.com/questions/tagged/sktime)

## Modules Overview

| Module                             | Status    | Links                                                                                                                                                                                                                                                                                                                                        |
| ---------------------------------- | --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Forecasting**                    | stable    | [Tutorial](https://www.sktime.net/en/latest/examples/01_forecasting.html) · [API Reference](https://www.sktime.net/en/latest/api_reference/forecasting.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/forecasting.py)                                                                  |
| **Time Series Classification**       | stable    | [Tutorial](https://github.com/sktime/sktime/blob/main/examples/02_classification.ipynb) · [API Reference](https://www.sktime.net/en/latest/api_reference/classification.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/classification.py)                                                      |
| **Time Series Regression**           | stable    | [API Reference](https://www.sktime.net/en/latest/api_reference/regression.html)                                                                                                                                                                                                                                                              |
| **Transformations**                | stable    | [Tutorial](https://github.com/sktime/sktime/blob/main/examples/03_transformers.ipynb) · [API Reference](https://www.sktime.net/en/latest/api_reference/transformations.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/transformer.py)                                                                  |
| **Detection tasks**                | maturing  | [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/detection.py)                                                                                                                                                                                                                                             |
| **Parameter fitting**              | maturing  | [API Reference](https://www.sktime.net/en/latest/api_reference/param_est.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/transformer.py)                                                                                                                                                     |
| **Time Series Clustering**         | maturing  | [API Reference](https://www.sktime.net/en/latest/api_reference/clustering.html) ·  [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/clustering.py)                                                                                                                                                |
| **Time Series Distances/Kernels**  | maturing  | [Tutorial](https://github.com/sktime/sktime/blob/main/examples/03_transformers.ipynb) · [API Reference](https://www.sktime.net/en/latest/api_reference/dists_kernels.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/dist_kern_panel.py)                                                    |
| **Time Series Alignment**          | experimental | [API Reference](https://www.sktime.net/en/latest/api_reference/alignment.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/alignment.py)                                                                                                                                                    |
| **Time Series Splitters**          | maturing  | [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/split.py)                                                                                                                                                                                                                                            |
| **Distributions and simulation** | experimental |  |

## Installation

*   **Prerequisites:** macOS X, Linux, Windows 8.1 or higher, Python 3.8+ (64-bit only).
*   **Package Managers:** `pip` and `conda` (via `conda-forge`).

### pip

```bash
pip install sktime
```

or with all extras:

```bash
pip install sktime[all_extras]
```

### conda

```bash
conda install -c conda-forge sktime
```

## Quickstart

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

*   **[Contribute](https://www.sktime.net/en/latest/get_involved/contributing.html):** Learn how to contribute to sktime.
*   **[Mentoring](https://github.com/sktime/mentoring):** Apply to our mentoring program.
*   **[Meetings](https://calendar.google.com/calendar/u/0/embed?src=sktime.toolbox@gmail.com&ctz=UTC):** Join our discussions, tutorials, workshops, and sprints!
*   **[Developer Guides](https://www.sktime.net/en/latest/developer_guide.html):** Learn how to develop sktime's code base.
*   **[Enhancement Proposals](https://github.com/sktime/enhancement-proposals):** Design a new feature for sktime.
*   **[Contributors](https://github.com/sktime/sktime/blob/main/CONTRIBUTORS.md):** A list of all contributors.
*   **[Roles](https://www.sktime.net/en/latest/about/team.html):** An overview of our core community roles.
*   **[Donate](https://opencollective.com/sktime):** Support sktime development.
*   **[Governance](https://www.sktime.net/en/latest/get_involved/governance.html):** How decisions are made in sktime.

## Project Vision

*   **Community-Driven:** Developed by a friendly and collaborative community.
*   **Task-Oriented:** Helps users choose the right tool for their time series task.
*   **Ecosystem Integration:** Interoperable with leading time series and machine learning libraries.
*   **Comprehensive Functionality:** Enables building, tuning, and analyzing time series models.
*   **User-Friendly:** Based on modern object-oriented design principles.
*   **Robust Assessment:** Provides tools for fair model evaluation and benchmarking.
*   **Extensible:** Designed to easily incorporate new algorithms.