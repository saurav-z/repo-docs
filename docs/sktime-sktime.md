# sktime: Your Unified Toolkit for Time Series Analysis

<a href="https://www.sktime.net"><img src="https://github.com/sktime/sktime/blob/main/docs/source/images/sktime-logo.svg" width="175" align="right" /></a>

**sktime is a comprehensive Python library offering a unified interface for all your time series machine learning needs.** This includes forecasting, classification, clustering, anomaly detection, and more.  [Visit our GitHub repository](https://github.com/sktime/sktime) to explore the code and contribute!

**Key Features:**

*   **Unified Interface:** Consistent API for various time series tasks.
*   **Broad Functionality:** Covers forecasting, classification, clustering, and anomaly detection.
*   **Rich Algorithm Selection:**  Includes a wide range of time series algorithms.
*   **Scikit-learn Compatibility:** Integrates seamlessly with scikit-learn tools for model building, tuning, and validation.
*   **Extensibility:** Easy-to-use templates for adding custom algorithms.
*   **Integration:** Interfaces with popular libraries like scikit-learn, statsmodels, and others.

**Version 0.38.4 is now available!** [View the release notes](https://www.sktime.net/en/latest/changelog.html) to see what's new.

## Core Functionality

sktime provides comprehensive support for various time series analysis tasks:

*   **Forecasting:** Predict future values of a time series.
*   **Time Series Classification:** Categorize time series data.
*   **Time Series Regression:** Predict continuous values from time series data.
*   **Transformations:**  Apply data transformations for analysis.
*   **Anomaly/Change Point Detection:** Identify unusual patterns in time series.
*   **Clustering:** Group similar time series together.
*   **Distances/Kernels:** Measure similarity between time series.

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

## Documentation & Resources

*   [Documentation](https://www.sktime.net/en/stable/users.html)
*   [Tutorials](https://www.sktime.net/en/stable/examples.html)
*   [Release Notes](https://www.sktime.net/en/stable/changelog.html)

## Get Involved

Join the sktime community and contribute!

*   [Contributing Guidelines](https://www.sktime.net/en/latest/get_involved/contributing.html)
*   [GitHub Discussions](https://github.com/sktime/sktime/discussions)
*   [Discord](https://discord.com/invite/54ACzaFsn7)

## Installation

Install sktime using pip or conda:

**Pip:**

```bash
pip install sktime
```

or with all extras:

```bash
pip install sktime[all_extras]
```

**Conda:**

```bash
conda install -c conda-forge sktime
```

## Project Vision

*   **Community-Driven:** Developed by and for the community.
*   **Task-Oriented:** The right tool for the right time series task.
*   **Interoperable:** Works with other popular data science libraries.
*   **Model Composition:** Features tools for model building, tuning, and feature extraction.
*   **Clear Syntax:** Based on modern object-oriented design principles.
*   **Fair Assessment:** Tools for model evaluation and benchmarking.
*   **Extensible:** Allows for easy addition of custom algorithms.