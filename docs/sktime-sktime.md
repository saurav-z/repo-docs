# sktime: Your Unified Toolkit for Time Series Analysis

[![sktime Logo](https://github.com/sktime/sktime/blob/main/docs/source/images/sktime-logo.svg?raw=true)](https://www.sktime.net)

**sktime is a powerful Python library that simplifies time series analysis by providing a unified interface for various machine-learning tasks.** ([Original Repository](https://github.com/sktime/sktime))

*   :rocket: **Version 0.38.4 is out now!** [Check out the release notes here](https://www.sktime.net/en/latest/changelog.html).

sktime offers a comprehensive suite of tools for time series analysis, including forecasting, classification, clustering, anomaly detection, and more. It provides a consistent API and integrates seamlessly with popular libraries like scikit-learn, enabling you to build, tune, and validate time series models efficiently.

## Key Features

*   **Unified Interface:** A consistent API for diverse time series tasks, streamlining your workflow.
*   **Wide Range of Tasks:** Support for forecasting, classification, regression, clustering, anomaly detection, and more.
*   **Built-in Algorithms:** Includes a variety of time series algorithms ready to use.
*   **scikit-learn Compatibility:** Leverages scikit-learn tools for model building, tuning, and evaluation.
*   **Extensibility:** Easy to extend with custom algorithms, providing flexibility for advanced use cases.
*   **Interoperability:** Interfaces with popular libraries like scikit-learn, statsmodels, tsfresh, PyOD, and fbprophet.

## Documentation & Resources

*   **[Documentation](https://www.sktime.net/en/stable/users.html)**
*   **[Tutorials](https://www.sktime.net/en/latest/tutorials.html)**
*   **[Examples](https://www.sktime.net/en/latest/examples.html)**
*   **[Release Notes](https://www.sktime.net/en/latest/changelog.html)**

### Comprehensive Documentation

*   **[Tutorials]:** Get started with sktime!
*   **[Binder Notebooks]:** Interactive examples to play with.
*   **[Examples]:** Learn how to use sktime features.
*   **[API Reference]:** Detailed API documentation.
*   **[Changelog]:** Changes and version history.

## Get Involved

*   **[Contribute]:** Learn how to contribute to sktime.
*   **[GitHub Discussions]:** Ask questions and share ideas.
*   **[Discord]:** Join the community chat.

## Installation

Install sktime using pip or conda.  See [Installation](https://www.sktime.net/en/latest/installation.html) for detailed instructions.

**pip:**

```bash
pip install sktime
```

**conda:**

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

## Project Vision

*   **Community-Driven:** Developed by a collaborative and friendly community.
*   **Task-Oriented:** Assists users in identifying the right tools for their machine-learning problems.
*   **Interoperable:** Integrates with popular ecosystems like scikit-learn, statsmodels, and others.
*   **Feature-Rich:** Provides model composition, reduction, and tuning functionality.
*   **User-Friendly:** Uses clean and descriptive syntax based on modern design principles.
*   **Robust Evaluation:** Supports fair model assessment and benchmarking.
*   **Extensible:** Offers easy extension templates for adding custom algorithms.