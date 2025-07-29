# sktime: Unified Time Series Analysis in Python

[![sktime Logo](https://github.com/sktime/sktime/blob/main/docs/source/images/sktime-logo.svg?raw=true)](https://www.sktime.net)

**sktime is your one-stop Python library for comprehensive time series analysis, offering a unified interface for forecasting, classification, and more.**

[Check out the original repository](https://github.com/sktime/sktime)

**Key Features:**

*   ✅ **Unified Interface:** Simplify time series analysis with a consistent API for various tasks.
*   ✅ **Forecasting:** Powerful tools for time series forecasting, including models and evaluation metrics.
*   ✅ **Classification & Regression:** Implement time series classification and regression models.
*   ✅ **Clustering:** Explore time series clustering algorithms for pattern discovery.
*   ✅ **Anomaly Detection:** Identify unusual patterns and events within your time series data.
*   ✅ **Transformations:** Apply a wide range of time series transformations.
*   ✅ **Extensible:** Easy to extend with custom algorithms using sktime's API.
*   ✅ **Integration:** Seamlessly integrates with popular libraries like scikit-learn, statsmodels, and more.

**Key Features:**
*   ✅ **Unified Interface:** Simplify time series analysis with a consistent API for various tasks.
*   ✅ **Forecasting:** Powerful tools for time series forecasting, including models and evaluation metrics.
*   ✅ **Classification & Regression:** Implement time series classification and regression models.
*   ✅ **Clustering:** Explore time series clustering algorithms for pattern discovery.
*   ✅ **Anomaly Detection:** Identify unusual patterns and events within your time series data.
*   ✅ **Transformations:** Apply a wide range of time series transformations.
*   ✅ **Extensible:** Easy to extend with custom algorithms using sktime's API.
*   ✅ **Integration:** Seamlessly integrates with popular libraries like scikit-learn, statsmodels, and more.

## Core Modules & Functionality

*   **Forecasting:** Stable, ready-to-use forecasting models and tools.
*   **Time Series Classification:** Robust methods for classifying time series data.
*   **Time Series Regression:** Perform regression tasks with time series data.
*   **Transformations:** Extensive collection of transformation methods.
*   **Anomaly/Change Point Detection:** Identify anomalies and change points in time series data.
*   **Clustering:** Discover patterns and group similar time series.
*   **Distance Metrics & Kernels:** Calculate distances and kernels for time series data.
*   **Time Series Alignment:** Align and compare time series.

## Documentation & Resources

*   **[Documentation](https://www.sktime.net/en/stable/users.html)**: Comprehensive user guide and API reference.
*   **[Tutorials](https://www.sktime.net/en/latest/tutorials.html)**: Step-by-step tutorials to get you started.
*   **[Examples](https://www.sktime.net/en/latest/examples.html)**: Practical examples demonstrating sktime features.
*   **[API Reference](https://www.sktime.net/en/latest/api_reference.html)**: Detailed API reference for all sktime components.
*   **[Changelog](https://www.sktime.net/en/latest/changelog.html)**: Keep up-to-date with the latest changes and releases.

## Getting Involved

*   **[Contribute](https://www.sktime.net/en/latest/get_involved/contributing.html)**: Learn how to contribute to the sktime project.
*   **[Community](https://discord.com/invite/54ACzaFsn7)**: Join the sktime community on Discord.

## Installation

```bash
pip install sktime
```

Or, to install with extra dependencies for various tasks:

```bash
pip install sktime[all_extras]
```

## Quickstart: Forecasting

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

## Quickstart: Time Series Classification

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

## License
[BSD 3-clause](https://github.com/sktime/sktime/blob/main/LICENSE)

## Funding
[Sponsored Project](https://gc-os-ai.github.io/)