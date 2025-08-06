# sktime: Your All-in-One Library for Time Series Analysis

[![sktime Logo](https://github.com/sktime/sktime/blob/main/docs/source/images/sktime-logo.svg?raw=true)](https://www.sktime.net)

**sktime is a unified Python library that simplifies time series analysis, offering a comprehensive suite of tools for various tasks.**

*   ✅ **Unified Interface:** Consistent API for diverse time series tasks.
*   ✅ **Forecasting:** Powerful forecasting algorithms and tools.
*   ✅ **Classification:** State-of-the-art time series classification methods.
*   ✅ **Clustering:**  Effective time series clustering techniques.
*   ✅ **Anomaly & Change Point Detection:** Robust methods for identifying anomalies.
*   ✅ **Integration:** Seamless integration with scikit-learn and other popular libraries.

[Explore the sktime repository on GitHub](https://github.com/sktime/sktime)

**🚀 Version 0.38.4 is now available!** [Read the release notes here](https://www.sktime.net/en/latest/changelog.html).

## Key Features

sktime provides a unified interface for various time series learning tasks. It features dedicated time series algorithms and tools for composite model building, such as pipelining, ensembling, tuning, and reduction, empowering users to apply algorithms designed for one task to another.

*   **Time Series Forecasting:**  Includes various forecasting models and tools.
*   **Time Series Classification:** Offers a range of classification algorithms.
*   **Time Series Clustering:** Supports different clustering techniques.
*   **Anomaly Detection:** Provides methods for identifying unusual patterns.
*   **Model Composition:** Allows building, tuning, and validating time series models.
*   **Integration with Existing Tools:** Compatible with scikit-learn, statsmodels, and more.

## Documentation and Resources

Access comprehensive guides, tutorials, and examples to get started with sktime:

*   [Documentation](https://www.sktime.net/en/stable/users.html)
*   [Tutorials](https://www.sktime.net/en/latest/tutorials.html)
*   [Examples](https://www.sktime.net/en/latest/examples.html)
*   [API Reference](https://www.sktime.net/en/latest/api_reference.html)
*   [Changelog](https://www.sktime.net/en/latest/changelog.html)
*   [Roadmap](https://www.sktime.net/en/latest/roadmap.html)
*   [Related Software](https://www.sktime.net/en/latest/related_software.html)

## Get Involved

Join the sktime community and contribute to its development.  We welcome all types of contributions.

*   [Contribute to sktime](https://www.sktime.net/en/latest/get_involved/contributing.html)
*   [Join the Discord Community](https://discord.com/invite/54ACzaFsn7)
*   [GitHub Issue Tracker](https://github.com/sktime/sktime/issues)
*   [GitHub Discussions](https://github.com/sktime/sktime/discussions)
*   [Stack Overflow](https://stackoverflow.com/questions/tagged/sktime)
*   [Mentoring Program](https://github.com/sktime/mentoring)
*   [Meetings](https://calendar.google.com/calendar/u/0/embed?src=sktime.toolbox@gmail.com&ctz=UTC)

## Installation

Install sktime using pip or conda:

### pip

```bash
pip install sktime
```

or install all dependencies

```bash
pip install sktime[all_extras]
```

### conda

```bash
conda install -c conda-forge sktime
```

or install all dependencies

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

## Project Vision

*   **Community-Driven:** Developed by a collaborative and friendly community.
*   **Task-Oriented:** Helps users select the best models for their learning problems.
*   **Interoperable:** Compatible with scikit-learn, statsmodels, and other libraries.
*   **Model Composition:** Supports building pipelines, tuning, and feature extraction.
*   **Clean Syntax:** Based on modern object-oriented design principles.
*   **Fair Evaluation:** Encourages best practices in model assessment and benchmarking.
*   **Extensible:** Easy-to-use extension templates for adding custom algorithms.