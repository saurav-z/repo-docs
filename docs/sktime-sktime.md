<!-- README.md -->

# sktime: Your Unified Toolkit for Time Series Analysis

<a href="https://www.sktime.net">
    <img src="https://github.com/sktime/sktime/blob/main/docs/source/images/sktime-logo.svg" width="175" align="right" alt="sktime logo"/>
</a>

**sktime is a comprehensive Python library providing a unified interface for all your time series analysis needs.**  Whether you're forecasting, classifying, clustering, or detecting anomalies, sktime offers a user-friendly and extensible framework. [Explore the sktime repository](https://github.com/sktime/sktime).

*   **Unified Interface:** Simplify your time series analysis workflow with a consistent API.
*   **Diverse Tasks:**  Supports forecasting, classification, clustering, anomaly detection, and more.
*   **Extensive Algorithms:** Includes built-in time series algorithms and seamless integration with scikit-learn tools.
*   **Integration:** Connects with popular libraries like scikit-learn, statsmodels, tsfresh, PyOD, and Prophet.
*   **Community-Driven:** Benefit from a vibrant open-source community.

**Latest Release:** [Version 0.38.4](https://www.sktime.net/en/latest/changelog.html)

## Key Features

*   **Forecasting:**  Predict future time series values.
*   **Time Series Classification:** Categorize time series data.
*   **Time Series Clustering:** Group similar time series together.
*   **Anomaly/Change Point Detection:** Identify unusual patterns.
*   **Transformation and Feature Extraction:**  Prepare your data for analysis.
*   **Model Pipelines:**  Build complex, modular time series models.
*   **Extensibility:** Easily integrate custom algorithms.

## Getting Started

### Installation

Install sktime using pip or conda:

```bash
pip install sktime
```

or, with maximum dependencies:

```bash
pip install sktime[all_extras]
```

For curated sets of soft dependencies for specific learning tasks:

```bash
pip install sktime[forecasting]  # for selected forecasting dependencies
pip install sktime[forecasting,transformations]  # forecasters and transformers
```

or similar. Valid sets are:

* `forecasting`
* `transformations`
* `classification`
* `regression`
* `clustering`
* `param_est`
* `networks`
* `detection`
* `alignment`

**Conda Install (via conda-forge):**

```bash
conda install -c conda-forge sktime
```

or, with maximum dependencies:

```bash
conda install -c conda-forge sktime-all-extras
```

### Quickstart

Here's a glimpse of sktime in action:

**Forecasting Example**

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

**Time Series Classification Example**

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

## Documentation and Resources

*   [Documentation](https://www.sktime.net/en/stable/users.html)
*   [Tutorials](https://www.sktime.net/en/stable/examples.html)
*   [Release Notes](https://www.sktime.net/en/latest/changelog.html)

## Community and Support

*   **GitHub Discussions:** [https://github.com/sktime/sktime/discussions](https://github.com/sktime/sktime/discussions)
*   **Discord:** [https://discord.com/invite/54ACzaFsn7](https://discord.com/invite/54ACzaFsn7)
*   **Stack Overflow:** [https://stackoverflow.com/questions/tagged/sktime](https://stackoverflow.com/questions/tagged/sktime)

## How to Contribute

Join the sktime community and help shape the future of time series analysis! We welcome contributions of all kinds.

*   [Contribute](https://www.sktime.net/en/latest/get_involved/contributing.html)
*   [Mentoring](https://github.com/sktime/mentoring)
*   [Meetings](https://calendar.google.com/calendar/u/0/embed?src=sktime.toolbox@gmail.com&ctz=UTC)
*   [Developer Guides](https://www.sktime.net/en/latest/developer_guide.html)
*   [Enhancement Proposals](https://github.com/sktime/enhancement-proposals)
*   [Roles](https://www.sktime.net/en/latest/about/team.html)

## Project Vision

*   **Community-Driven:**  Developed by and for the community.
*   **Task-Oriented:** Provides the right tools for the right analysis.
*   **Interoperable:** Works seamlessly with popular data science ecosystems.
*   **Modular and Extensible:** Easy to build and extend models.
*   **Fair and Transparent:**  Prioritizes model interpretability and robust evaluation.

##  Hall of Fame

A big thank you to all our contributors!

<a href="https://github.com/sktime/sktime/graphs/contributors">
<img src="https://opencollective.com/sktime/contributors.svg?width=600&button=false" alt="Contributors"/>
</a>