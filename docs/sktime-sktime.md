# sktime: Your Unified Toolkit for Time Series Analysis

<a href="https://www.sktime.net"><img src="https://github.com/sktime/sktime/blob/main/docs/source/images/sktime-logo.svg" width="175" align="right" /></a>

**sktime is a Python library providing a unified interface for all your time series machine learning needs, including forecasting, classification, and more.**  This open-source project offers a comprehensive suite of tools for time series analysis, including algorithms, model building, and evaluation.

**[Visit the sktime Repository on GitHub](https://github.com/sktime/sktime)**

**Key Features:**

*   **Unified Interface:** Simplifies time series learning tasks with a consistent API.
*   **Diverse Algorithms:** Includes a wide range of time series algorithms for forecasting, classification, clustering, anomaly detection, and more.
*   **Model Building & Tuning:**  Provides tools for building, tuning, and validating time series models, including pipelining and ensembling.
*   **Integration:**  Compatible with [scikit-learn], [statsmodels], [tsfresh], and other popular Python libraries.
*   **Extensibility:**  Easy-to-use extension templates for adding your own algorithms.

**What's New:** :rocket: **Version 0.38.4 is now available!** [See the release notes](https://www.sktime.net/en/latest/changelog.html)

## Documentation

*   **[Tutorials](https://www.sktime.net/en/latest/tutorials.html)**: Start here if you are new to sktime!
*   **[Examples](https://www.sktime.net/en/latest/examples.html)**: How to use sktime and its features.
*   **[API Reference](https://www.sktime.net/en/latest/api_reference.html)**: The detailed reference for sktime's API.
*   **[Changelog](https://www.sktime.net/en/latest/changelog.html)**: Changes and version history.
*   **[Roadmap](https://www.sktime.net/en/latest/roadmap.html)**: sktime's software and community development plan.
*   **[Related Software](https://www.sktime.net/en/latest/related_software.html)**: A list of related software.

## Key Modules & Functionality

*   **Forecasting:**  Stable forecasting module with tutorials and API reference.
*   **Time Series Classification:** Stable module with tutorials and extension templates.
*   **Time Series Regression:** Stable module with API reference.
*   **Transformations:** Stable module for time series transformations with tutorials and API reference.
*   **Detection tasks:** Maturing detection task module with extension templates.
*   **Parameter fitting:** Maturing parameter fitting module with API reference.
*   **Time Series Clustering:** Maturing clustering module with API reference.
*   **Time Series Distances/Kernels:** Maturing module for time series distances/kernels with tutorials and API reference.
*   **Time Series Alignment:** Experimental module for time series alignment with API reference.
*   **Time Series Splitters:** Maturing module for time series splitters.
*   **Distributions and simulation:** Experimental module with distributions and simulation.

## Get Involved

*   **[Contribute](https://www.sktime.net/en/latest/get_involved/contributing.html)**: Learn how to contribute to sktime.
*   **[GitHub Discussions](https://github.com/sktime/sktime/discussions)**: Join the discussions!
*   **[Discord](https://discord.com/invite/54ACzaFsn7)**: Connect with the sktime community.

## Installation

*   **Python Versions:**  3.8, 3.9, 3.10, 3.11, and 3.12 (64-bit only)
*   **Package Managers:** `pip` and `conda` (via `conda-forge`)

### pip

```bash
pip install sktime
```

or, for all extras:

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