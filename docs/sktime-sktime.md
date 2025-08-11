# sktime: Unified Time Series Analysis in Python

**Analyze and model time series data effortlessly with sktime, a comprehensive Python library offering a unified interface for diverse time series tasks.**  ([sktime GitHub Repository](https://github.com/sktime/sktime))

[![sktime Logo](https://github.com/sktime/sktime/blob/main/docs/source/images/sktime-logo.svg)](https://www.sktime.net)

**Key Features:**

*   **Unified Interface:** Streamlines time series analysis by providing a consistent API for various tasks.
*   **Comprehensive Tasks:** Supports forecasting, time series classification, clustering, anomaly detection, and more.
*   **Extensive Algorithms:** Includes a wide range of time series algorithms.
*   **Scikit-learn Compatibility:** Seamlessly integrates with scikit-learn tools for model building, tuning, and evaluation.
*   **Interoperability:** Works with related libraries like scikit-learn, statsmodels, tsfresh, and PyOD.
*   **Easy Extensibility:** Offers extension templates to add custom algorithms easily.
*   **Community-Driven:** Developed by a collaborative community, fostering open-source contributions.

**What's New:** [Check out the latest release notes](https://www.sktime.net/en/latest/changelog.html)

## Core Functionality

*   **Forecasting:** Build, train and evaluate time series forecasts.
*   **Time Series Classification:** Easily classify time series data.
*   **Time Series Regression:** Regression capabilities for time series data.
*   **Transformations:** Perform time series data transformations.
*   **Clustering:** Group time series data for pattern discovery.
*   **Detection Tasks:**  Detect anomalies, change points and other features in time series.
*   **Parameter fitting**:  Parameter fitting for various time series models.
*   **Time Series Distances/Kernels**: Compute various distances and kernels on time series data.
*   **Time Series Alignment**: Align and compare time series data.
*   **Time Series Splitters**: Split time series data for model evaluation.

## Key Resources

*   **[Documentation](https://www.sktime.net/en/stable/users.html)**
*   **[Tutorials](https://www.sktime.net/en/stable/examples.html)**
*   **[Release Notes](https://www.sktime.net/en/latest/changelog.html)**

## Installation

Install sktime using pip or conda:

**pip:**

```bash
pip install sktime
```

**conda:**

```bash
conda install -c conda-forge sktime
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

## Get Involved

Join the sktime community! We welcome all contributions.

*   **[Contribute](https://www.sktime.net/en/latest/get_involved/contributing.html)**
*   **[Community](https://discord.com/invite/54ACzaFsn7)**
*   **[Discussions](https://github.com/sktime/sktime/discussions)**
*   **[Issue Tracker](https://github.com/sktime/sktime/issues)**

## Project Vision

*   **Community-driven:** Developed by and for the community.
*   **Problem-focused:** Helps users choose the right model for the task.
*   **Interoperable:**  Works well with existing data science tools.
*   **Model Composition:**  Offers extensive model composition and reduction.
*   **Clean Design:** Based on modern object-oriented design principles.
*   **Fair Assessment:**  Ensures proper model assessment and benchmarking.
*   **Extensible:** Easy to add your own algorithms and build extensions.