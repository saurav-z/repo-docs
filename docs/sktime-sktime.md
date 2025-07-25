# sktime: The Unified Toolkit for Time Series Machine Learning

[![sktime Logo](https://github.com/sktime/sktime/blob/main/docs/source/images/sktime-logo.svg?raw=true "sktime")](https://www.sktime.net/)

**sktime is a powerful Python library providing a unified interface for all your time series analysis needs, from forecasting to classification and beyond!**  ([See the original repo](https://github.com/sktime/sktime))

**Key Features:**

*   **Comprehensive Time Series Tasks:**
    *   Forecasting
    *   Time Series Classification
    *   Time Series Regression
    *   Clustering
    *   Anomaly/Changepoint Detection
    *   Time Series Transformation, Distances & Alignment
*   **Unified Interface:** Provides a consistent and intuitive API across various time series learning tasks.
*   **Extensive Algorithm Support:** Includes a wide range of built-in time series algorithms.
*   **Scikit-learn Compatibility:**  Integrates seamlessly with scikit-learn for familiar model building, tuning, and validation workflows.
*   **Interoperability:**  Connects with popular libraries like scikit-learn, statsmodels, tsfresh, and more.
*   **Model Composition and Reduction:** Features tools for pipelining, ensembling, tuning, and feature extraction.
*   **Easy Extensibility:**  Offers extension templates to add your own algorithms.
*   **Well-Documented:** Comprehensive documentation, tutorials, and examples to get you started.

**Version:** 0.38.4 ([Release Notes](https://www.sktime.net/en/latest/changelog.html))

**Key Resources:**

*   [Documentation](https://www.sktime.net/en/stable/users.html)
*   [Tutorials](https://www.sktime.net/en/stable/examples.html)
*   [API Reference](https://www.sktime.net/en/latest/api_reference.html)
*   [Release Notes](https://www.sktime.net/en/latest/changelog.html)

---

## What is sktime?

sktime is a Python library designed to simplify and standardize time series analysis. It provides a unified interface, making it easier to work with diverse time series learning tasks and algorithms.

---

## :books: Documentation

*   **[Tutorials](https://www.sktime.net/en/latest/tutorials.html):** Get started with sktime.
*   **[Binder Notebooks](https://mybinder.org/v2/gh/sktime/sktime/main?filepath=examples):** Interactive examples to play with in your browser.
*   **[Examples](https://www.sktime.net/en/latest/examples.html):**  Learn how to use sktime and its features through practical examples.
*   **[API Reference](https://www.sktime.net/en/latest/api_reference.html):** Detailed documentation of the sktime API.
*   **[Changelog](https://www.sktime.net/en/latest/changelog.html):** View changes and version history.
*   **[Roadmap](https://www.sktime.net/en/latest/roadmap.html):** sktime's development plan.

---

## :speech_balloon: Get Involved & Ask Questions

*   **Bug Reports & Feature Requests:** [GitHub Issue Tracker](https://github.com/sktime/sktime/issues)
*   **Usage Questions & General Discussion:** [GitHub Discussions](https://github.com/sktime/sktime/discussions)
*   **Development & Contribution:**  `dev-chat` channel on Discord, [Discord](https://discord.com/invite/54ACzaFsn7)

---

## :dizzy: Core Features & Modules

sktime facilitates a unified interface for diverse time series learning tasks. The following are the core areas:

*   **Forecasting:** Stable, with a [Tutorial](https://www.sktime.net/en/latest/examples/01_forecasting.html)
*   **Time Series Classification:** Stable, with a [Tutorial](https://github.com/sktime/sktime/blob/main/examples/02_classification.ipynb)
*   **Time Series Regression:** Stable
*   **Transformations:** Stable, with a [Tutorial](https://github.com/sktime/sktime/blob/main/examples/03_transformers.ipynb)
*   **Detection Tasks:** Maturing
*   **Time Series Clustering:** Maturing
*   **Time Series Distances/Kernels:** Maturing, with a [Tutorial](https://github.com/sktime/sktime/blob/main/examples/03_transformers.ipynb)
*   **Time Series Alignment:** Experimental

---

## :hourglass_flowing_sand: Installation

**Prerequisites:** macOS X, Linux, Windows 8.1 or higher with Python 3.8, 3.9, 3.10, 3.11, and 3.12 (64-bit only).

**Installation via pip:**

```bash
pip install sktime
```

For maximum dependencies:

```bash
pip install sktime[all_extras]
```

For task-specific dependencies:

```bash
pip install sktime[forecasting]  # For forecasting dependencies
pip install sktime[forecasting,transformations]  # Forecasters and transformers
```

Valid sets are: `forecasting`, `transformations`, `classification`, `regression`, `clustering`, `param_est`, `networks`, `detection`, and `alignment`.

**Installation via conda:**

```bash
conda install -c conda-forge sktime
```

Or, with maximum dependencies:

```bash
conda install -c conda-forge sktime-all-extras
```

---

## :zap: Quickstart Examples

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

---

## :wave: Get Involved

*   [Contribute](https://www.sktime.net/en/latest/get_involved/contributing.html)
*   [Mentoring](https://github.com/sktime/mentoring)
*   [Meetings](https://calendar.google.com/calendar/u/0/embed?src=sktime.toolbox@gmail.com&ctz=UTC)
*   [Developer Guides](https://www.sktime.net/en/latest/developer_guide.html)
*   [Enhancement Proposals](https://github.com/sktime/enhancement-proposals)
*   [Contributors](https://github.com/sktime/sktime/blob/main/CONTRIBUTORS.md)
*   [Roles](https://www.sktime.net/en/latest/about/team.html)
*   [Donate](https://opencollective.com/sktime)
*   [Governance](https://www.sktime.net/en/latest/get_involved/governance.html)

---

## :trophy: Hall of Fame

A list of all the contributors can be found here:

<a href="https://github.com/sktime/sktime/graphs/contributors">
<img src="https://opencollective.com/sktime/contributors.svg?width=600&button=false" />
</a>

---

## :bulb: Project Vision

*   **Community-Driven:** Developed by a friendly and collaborative community.
*   **Task-Specific:** The right tool for the right task to guide users to select an appropriate model.
*   **Interoperable:** Compatible with popular libraries like scikit-learn, statsmodels, and tsfresh.
*   **Rich Model Composition:** Features for building tuning and feature extraction pipelines.
*   **Clean Syntax:** Based on modern object-oriented design principles.
*   **Fair Assessment:** Build, inspect, and validate models to avoid pitfalls.
*   **Extensible:** Easy extension templates.