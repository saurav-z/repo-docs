<div align="center">
  <a href="https://www.sktime.net">
    <img src="https://github.com/sktime/sktime/blob/main/docs/source/images/sktime-logo.svg" alt="sktime Logo" width="200">
  </a>
  <h1>sktime: Time Series Analysis Made Easy</h1>
  <p><em>Unlock the power of time series data with sktime, a unified and user-friendly Python library.</em></p>
  <p>
    <a href="https://www.sktime.net/en/latest/changelog.html">
      <img src="https://img.shields.io/badge/version-0.38.4-blue" alt="Version">
    </a>
    <a href="https://github.com/sktime/sktime">
      <img src="https://img.shields.io/github/stars/sktime/sktime?style=social" alt="GitHub Stars">
    </a>
  </p>
</div>

---

## What is sktime?

sktime is a Python library that provides a unified interface for machine learning with time series data. It offers a comprehensive toolkit for various time series tasks, including forecasting, classification, clustering, anomaly detection, and more. Designed for both researchers and practitioners, sktime streamlines the development, tuning, and validation of time series models.

**Key Features:**

*   **Unified Interface:**  A consistent API across diverse time series learning tasks, simplifying model selection and comparison.
*   **Diverse Task Support:** Solutions for forecasting, time series classification, regression, clustering, anomaly detection, and more.
*   **Algorithm Integration:** Includes numerous built-in time series algorithms and provides interfaces to popular libraries like scikit-learn, statsmodels, tsfresh, and PyOD.
*   **Model Building & Evaluation:**  Offers scikit-learn compatible tools for building, tuning, and validating time series models, including pipelining and ensembling.
*   **Extensibility:**  Easy-to-use extension templates for adding custom algorithms and models that integrate seamlessly with the sktime ecosystem.
*   **Comprehensive Documentation:**  Extensive documentation, tutorials, and examples to help you get started.

**Get Started:**

*   [Documentation](https://www.sktime.net/en/stable/users.html)
*   [Tutorials](https://www.sktime.net/en/stable/examples.html)
*   [Release Notes](https://www.sktime.net/en/stable/changelog.html)
*   **[View the Source Code on GitHub](https://github.com/sktime/sktime)**

---

##  Features Overview

sktime provides a comprehensive set of tools for various time series analysis tasks, all within a unified and intuitive framework.

*   **Forecasting:** Build accurate time series forecasts using a wide array of algorithms.
*   **Time Series Classification:** Classify time series data with ease, using various classification models.
*   **Time Series Regression:** Perform time series regression tasks to predict continuous values.
*   **Transformations:** Apply a variety of transformations to time series data for feature engineering and analysis.
*   **Anomaly/Change Point Detection:**  Identify unusual patterns or changes in your time series data.
*   **Time Series Clustering:** Group similar time series together using clustering techniques.
*   **Time Series Distance/Kernels:** Utilize diverse distance and kernel methods for comparing time series.
*   **Time Series Alignment:** Align time series for improved analysis and comparison.
*   **Time Series Splitters:** Create custom time series splitters for cross-validation and testing.
*   **Distributions and simulation:** Probabilistic forecasting, simulation, and related methods.
*   **Parameter fitting:** Tools for estimating model parameters for a variety of time series algorithms.

---

## Installation

Install sktime using pip or conda. For detailed instructions and troubleshooting, see the [installation documentation](https://www.sktime.net/en/latest/installation.html).

**Prerequisites:**

*   macOS X, Linux, or Windows 8.1+
*   Python 3.8, 3.9, 3.10, 3.11, or 3.12 (64-bit only)

**pip:**

```bash
pip install sktime
```

For maximum dependencies:

```bash
pip install sktime[all_extras]
```

Or, for specific learning tasks:

```bash
pip install sktime[forecasting]
pip install sktime[forecasting,transformations]
```

**conda:**

```bash
conda install -c conda-forge sktime
```

For maximum dependencies:

```bash
conda install -c conda-forge sktime-all-extras
```

---

## Quickstart Examples

Here are a few examples to quickly get you started with sktime.

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

## Get Involved & Community

The sktime community welcomes contributions of all types!

*   **[Contribute](https://www.sktime.net/en/latest/get_involved/contributing.html):**  Learn how to contribute code, documentation, and more.
*   **[Mentoring](https://github.com/sktime/mentoring):** Join our mentoring program if you are new to open source!
*   **[Meetings](https://calendar.google.com/calendar/u/0/embed?src=sktime.toolbox@gmail.com&ctz=UTC):** Participate in discussions, tutorials, workshops, and sprints.
*   **[Developer Guides](https://www.sktime.net/en/latest/developer_guide.html):** Learn how to contribute to the sktime codebase.
*   **[Enhancement Proposals](https://github.com/sktime/enhancement-proposals):** Suggest or design new features.
*   **[Contributors](https://github.com/sktime/sktime/blob/main/CONTRIBUTORS.md):** See a list of contributors.
*   **[Roles](https://www.sktime.net/en/latest/about/team.html):** Learn more about the sktime core community roles.
*   **[Donate](https://opencollective.com/sktime):** Support the project's maintenance and development.
*   **[Governance](https://www.sktime.net/en/latest/get_involved/governance.html):** Understand how decisions are made in the sktime community.

**Community Platforms:**

*   **GitHub Discussions:** [Discussions](https://github.com/sktime/sktime/discussions)
*   **Discord:** [Discord](https://discord.com/invite/54ACzaFsn7)
*   **Stack Overflow:** [Stack Overflow](https://stackoverflow.com/questions/tagged/sktime)

---

## Project Vision

*   **Community-Driven:** Developed by and for a friendly and collaborative community.
*   **Task-Focused:** The right tool for the right task, helping users choose suitable models.
*   **Interoperable:** Embedded in state-of-the-art ecosystems, interoperable with scikit-learn, statsmodels, tsfresh, and others.
*   **Versatile:** Rich model composition, reduction functionality, and feature extraction pipelines.
*   **Clear Syntax:**  Based on modern object-oriented design principles for data science.
*   **Robust Assessment:**  Fair model assessment and benchmarking to avoid common pitfalls.
*   **Extensible:**  Easy extension templates for adding custom algorithms.

---

## Acknowledgements

*   <a href="https://github.com/sktime/sktime/graphs/contributors">
    <img src="https://opencollective.com/sktime/contributors.svg?width=600&button=false" />
    </a>