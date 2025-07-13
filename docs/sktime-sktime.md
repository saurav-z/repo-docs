<a href="https://www.sktime.net"><img src="https://github.com/sktime/sktime/blob/main/docs/source/images/sktime-logo.svg" width="175" align="right" /></a>

# sktime: Your Unified Toolkit for Time Series Analysis

**sktime** is a Python library providing a unified interface for machine learning with time series, offering a comprehensive suite of tools for various time series tasks. ( [Back to the top](https://github.com/sktime/sktime) )

## Key Features

*   **Unified Interface:** Provides a consistent API for diverse time series learning tasks.
*   **Forecasting:** Offers a wide range of forecasting algorithms and tools.
*   **Classification:** Includes time series classification algorithms for pattern recognition.
*   **Clustering:** Supports time series clustering for grouping similar time series.
*   **Anomaly Detection:** Implements methods for identifying outliers and anomalies in time series data.
*   **Transformation and Feature Extraction:** Enables data preprocessing and feature engineering for improved model performance.
*   **Integration:** Works seamlessly with popular libraries like scikit-learn.
*   **Extensibility:** Allows users to easily extend the library with custom algorithms.
*   **Comprehensive Documentation:** Includes tutorials, examples, and API references for easy use.
*   **Community-Driven:** Developed and maintained by a vibrant and collaborative community.

## Core Functionality

*   **Forecasting:**  Build and evaluate time series forecasting models.
*   **Time Series Classification:**  Classify time series data.
*   **Time Series Regression:** Perform regression tasks on time series.
*   **Transformations:** Apply transformations to time series data.
*   **Detection Tasks:**  Identify anomalies and change points.
*   **Time Series Clustering:** Group similar time series data.
*   **Time Series Distances/Kernels:** Calculate distances and kernels for time series analysis.
*   **Time Series Alignment:** Align time series data.
*   **Time Series Splitters:** Split time series data for evaluation and model building.
*   **Distributions and Simulation:** Model probabilistic time series behavior.

## Get Started

**Installation:**

Install sktime using pip or conda.

```bash
pip install sktime
```

```bash
conda install -c conda-forge sktime
```

**Quickstart Examples:**

*   **Forecasting:**

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

*   **Time Series Classification:**

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

## Resources

*   **[Documentation](https://www.sktime.net/en/stable/users.html)**
*   **[Tutorials](https://www.sktime.net/en/stable/examples.html)**
*   **[Release Notes](https://www.sktime.net/en/stable/changelog.html)**
*   **[API Reference](https://www.sktime.net/en/latest/api_reference.html)**
*   **[Examples](https://www.sktime.net/en/latest/examples.html)**

## Get Involved

*   **[Contribute]** How to contribute to sktime.
*   **[Mentoring]** Apply to our mentoring program!
*   **[Meetings]** Join our discussions, tutorials, workshops, and sprints!
*   **[Developer Guides]** How to further develop sktime's code base.
*   **[Enhancement Proposals]** Design a new feature for sktime.
*   **[Contributors]** A list of all contributors.
*   **[Roles]** An overview of our core community roles.
*   **[Donate]** Fund sktime maintenance and development.
*   **[Governance]** How and by whom decisions are made in sktime's community.

## :bulb: Project vision

*   **By the community, for the community** -- developed by a friendly and collaborative community.
*   The **right tool for the right task** -- helping users to diagnose their learning problem and suitable scientific model types.
*   **Embedded in state-of-art ecosystems** and **provider of interoperable interfaces** -- interoperable with [scikit-learn], [statsmodels], [tsfresh], and other community favorites.
*   **Rich model composition and reduction functionality** -- build tuning and feature extraction pipelines, solve forecasting tasks with [scikit-learn] regressors.
*   **Clean, descriptive specification syntax** -- based on modern object-oriented design principles for data science.
*   **Fair model assessment and benchmarking** -- build your models, inspect your models, check your models, and avoid pitfalls.
*   **Easily extensible** -- easy extension templates to add your own algorithms compatible with sktime's API.

## :trophy: Hall of fame

Thanks to all our community for all your wonderful contributions, PRs, issues, ideas.

<a href="https://github.com/sktime/sktime/graphs/contributors">
<img src="https://opencollective.com/sktime/contributors.svg?width=600&button=false" />
</a>