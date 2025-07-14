# sktime: A Unified Python Library for Time Series Analysis

[![sktime Logo](https://github.com/sktime/sktime/blob/main/docs/source/images/sktime-logo.svg?raw=true)](https://www.sktime.net)

**Tackle time series challenges with ease using sktime, a comprehensive Python library offering a unified interface for various time series learning tasks.** ([Original Repository](https://github.com/sktime/sktime))

**Key Features:**

*   **Unified Interface:** Simplifies time series analysis by providing a consistent API for various tasks.
*   **Comprehensive Task Support:** Includes forecasting, classification, clustering, anomaly detection, and more.
*   **Versatile Algorithms:** Offers a wide range of time series algorithms.
*   **Scikit-learn Compatibility:** Integrates seamlessly with scikit-learn tools for model building, tuning, and validation.
*   **Extensive Documentation and Tutorials:** Provides thorough documentation, tutorials, and examples.
*   **Community-Driven:** Benefit from a vibrant and supportive community.

**What's New:**

*   :rocket: **Version 0.38.3 out now!** [Check out the release notes here](https://www.sktime.net/en/latest/changelog.html).

**Core Functionality and Key Modules:**

*   **Forecasting:** Predict future values of time series data.
*   **Time Series Classification:** Categorize time series data.
*   **Time Series Regression:** Predict continuous values from time series data.
*   **Transformations:** Modify time series data for improved analysis.
*   **Detection Tasks:** Identify anomalies and change points in time series data.
*   **Parameter Fitting:** Optimize model parameters.
*   **Time Series Clustering:** Group similar time series together.
*   **Time Series Distances/Kernels:** Measure the similarity between time series.
*   **Time Series Alignment:** Align time series data for comparison.
*   **Time Series Splitters:** Divide time series data for model evaluation.
*   **Distributions and Simulation:** Work with distributions and simulate time series data.

| Category             | Description                                                                  |
| :------------------- | :--------------------------------------------------------------------------- |
| **[Documentation](https://www.sktime.net/en/stable/users.html)**  | Comprehensive guides, tutorials, and API reference.   |
| **[Tutorials](https://www.sktime.net/en/stable/examples.html)**     | Step-by-step tutorials for getting started.           |
| **[Release Notes](https://www.sktime.net/en/stable/changelog.html)** | Stay up-to-date with the latest changes and features. |

**Resources & Community:**

*   **Documentation:**
    *   [Tutorials](https://www.sktime.net/en/latest/tutorials.html)
    *   [Binder Notebooks](https://mybinder.org/v2/gh/sktime/sktime/main?filepath=examples)
    *   [Examples](https://www.sktime.net/en/latest/examples.html)
    *   [API Reference](https://www.sktime.net/en/latest/api_reference.html)
    *   [Changelog](https://www.sktime.net/en/latest/changelog.html)
    *   [Roadmap](https://www.sktime.net/en/latest/roadmap.html)
    *   [Related Software](https://www.sktime.net/en/latest/related_software.html)
*   **Community:**
    *   [Discord](https://discord.com/invite/54ACzaFsn7)
    *   [LinkedIn](https://www.linkedin.com/company/scikit-time/)
*   **Code & Development:**
    *   [GitHub Actions](https://github.com/sktime/sktime/actions/workflows/wheels.yml)
    *   [Read the Docs](https://www.sktime.net/en/latest/?badge=latest)
    *   [PyPI](https://pypi.org/project/sktime/)
    *   [Conda](https://anaconda.org/conda-forge/sktime)
    *   [Zenodo](https://doi.org/10.5281/zenodo.3749000)
    *   [Python Versions](https://www.python.org/)
    *   [Black Code Style](https://github.com/psf/black)
*   **Downloads:**
    *   [PyPI Downloads](https://img.shields.io/pypi/dw/sktime)
    *   [PyPI Downloads](https://img.shields.io/pypi/dm/sktime)
    *   [Cumulative Downloads](https://pepy.tech/project/sktime)

**Installation:**

Detailed installation instructions are available in the [documentation](https://www.sktime.net/en/latest/installation.html).

*   **Python Versions:** 3.8, 3.9, 3.10, 3.11, and 3.12 (64-bit only)
*   **Package Managers:** pip and conda (via conda-forge)

```bash
# Using pip
pip install sktime
pip install sktime[all_extras] # for maximum dependencies
pip install sktime[forecasting]  # for selected forecasting dependencies
pip install sktime[forecasting,transformations]  # forecasters and transformers

# Using conda
conda install -c conda-forge sktime
conda install -c conda-forge sktime-all-extras
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

**Get Involved:**

Join the sktime community! All contributions are welcome.

*   [Contribute](https://www.sktime.net/en/latest/get_involved/contributing.html)
*   [Mentoring](https://github.com/sktime/mentoring)
*   [Meetings](https://calendar.google.com/calendar/u/0/embed?src=sktime.toolbox@gmail.com&ctz=UTC)
*   [Developer Guides](https://www.sktime.net/en/latest/developer_guide.html)
*   [Enhancement Proposals](https://github.com/sktime/enhancement-proposals)
*   [Contributors](https://github.com/sktime/sktime/blob/main/CONTRIBUTORS.md)
*   [Roles](https://www.sktime.net/en/latest/about/team.html)
*   [Donate](https://opencollective.com/sktime)
*   [Governance](https://www.sktime.net/en/latest/get_involved/governance.html)

**Hall of Fame:**

*   [Contributors](https://github.com/sktime/sktime/graphs/contributors)

**Project Vision:**

*   **Community-Driven:** Developed by a friendly and collaborative community.
*   **Task-Oriented:**  Helps users select the right tool for the right task.
*   **Interoperable:** Integrates with scikit-learn, statsmodels, and other popular libraries.
*   **Rich Functionality:**  Provides robust model composition, tuning, and feature extraction.
*   **Clean Syntax:** Based on modern object-oriented design principles.
*   **Fair Assessment:**  Supports fair model assessment and benchmarking.
*   **Extensible:** Easy extension templates to add your own algorithms compatible with sktime's API.