# sktime: Your Unified Toolkit for Time Series Analysis

**Unify and simplify your time series analysis with sktime, a comprehensive Python library offering a cohesive interface for diverse time series tasks.**

[<img src="https://github.com/sktime/sktime/blob/main/docs/source/images/sktime-logo.svg" width="175" align="right" />](https://github.com/sktime/sktime)

*   **Release:** Version 0.38.3 ([Changelog](https://www.sktime.net/en/latest/changelog.html))
*   **Source:** [GitHub Repository](https://github.com/sktime/sktime)

## Key Features

*   **Comprehensive Time Series Tasks:** Support for forecasting, classification, clustering, anomaly/changepoint detection, and more.
*   **Unified Interface:** Provides a consistent API for various time series learning tasks.
*   **Rich Algorithm Library:** Includes a wide range of built-in time series algorithms.
*   **scikit-learn Compatibility:** Integrates seamlessly with scikit-learn tools for building, tuning, and validating models.
*   **Model Composition Tools:** Features for pipelining, ensembling, tuning, and reduction, enabling advanced model building.
*   **Extensive Documentation:** Detailed documentation, tutorials, and examples to guide your analysis.

## Documentation & Resources

| Resource                         | Description                                                                   |
| -------------------------------- | ----------------------------------------------------------------------------- |
| [Documentation](https://www.sktime.net/en/stable/users.html) | Detailed user guides, API reference, and installation instructions.                                       |
| [Tutorials](https://www.sktime.net/en/latest/tutorials.html)   | Step-by-step guides to get you started with sktime.   |
| [Examples](https://www.sktime.net/en/latest/examples.html)   | Practical examples showcasing how to use sktime's features. |
| [Release Notes](https://www.sktime.net/en/latest/changelog.html)  | View the changes and version history.                                    |
| [Video Tutorial](https://www.youtube.com/playlist?list=PLKs3UgGjlWHqNzu0LEOeLKvnjvvest2d0)          | Our video tutorial from 2021 PyData Global.                                   |

## Core Functionality

*   **Forecasting:** Predict future values of a time series. ([Tutorial](https://www.sktime.net/en/latest/examples/01_forecasting.html), [API Reference](https://www.sktime.net/en/latest/api_reference/forecasting.html))
*   **Time Series Classification:** Categorize time series data. ([Tutorial](https://github.com/sktime/sktime/blob/main/examples/02_classification.ipynb), [API Reference](https://www.sktime.net/en/latest/api_reference/classification.html))
*   **Time Series Regression:** Predict continuous values based on time series data.
*   **Transformations:** Apply various transformations to time series data. ([Tutorial](https://github.com/sktime/sktime/blob/main/examples/03_transformers.ipynb), [API Reference](https://www.sktime.net/en/latest/api_reference/transformations.html))
*   **Detection Tasks:** Identify anomalies and change points in time series.
*   **Parameter Fitting:** Estimate parameters for time series models.
*   **Time Series Clustering:** Group similar time series together.
*   **Time Series Distances/Kernels:** Measure the similarity between time series.
*   **Time Series Alignment:** Align time series data for better comparison.

## Installation

Follow the instructions in the [documentation](https://www.sktime.net/en/latest/installation.html) to get started.

*   **Python:** 3.8, 3.9, 3.10, 3.11, and 3.12 (only 64-bit)
*   **Package Managers:** `pip` and `conda` (via `conda-forge`)

```bash
# Using pip
pip install sktime

# or, with maximum dependencies
pip install sktime[all_extras]

# Using conda
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

## Community & Contributing

Join the sktime community and contribute to the project!

*   [Contribute](https://www.sktime.net/en/latest/get_involved/contributing.html)
*   [GitHub Discussions](https://github.com/sktime/sktime/discussions)
*   [Discord](https://discord.com/invite/54ACzaFsn7)
*   [GitHub Issues](https://github.com/sktime/sktime/issues)

## Project Vision

*   **Community-Driven:** Developed by a collaborative and friendly community.
*   **Task-Oriented:** Helps users identify the right tools for their specific time series analysis tasks.
*   **Ecosystem Integration:** Interoperable with scikit-learn, statsmodels, tsfresh, and other popular libraries.
*   **Model Composition:** Offers rich functionality for building, tuning, and feature extraction pipelines.
*   **Clear Syntax:** Modern object-oriented design principles for data science.
*   **Fair Assessment:** Built-in tools for model inspection and evaluation.
*   **Extensible:** Easy extension templates for adding custom algorithms.