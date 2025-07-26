# sktime: Your Unified Toolkit for Time Series Analysis

[![sktime Logo](https://github.com/sktime/sktime/blob/main/docs/source/images/sktime-logo.svg?raw=true)](https://www.sktime.net)

**sktime is a Python library providing a unified interface for all your time series machine learning needs.** This versatile library offers a wide array of tools for time series analysis, including forecasting, classification, clustering, anomaly detection, and more. [Explore the full potential of sktime!](https://github.com/sktime/sktime)

*   🚀 **Key Features:**

    *   **Unified Interface:** A consistent API for various time series learning tasks.
    *   **Forecasting:** Robust forecasting algorithms.
    *   **Classification:** Powerful time series classification tools.
    *   **Clustering:** Advanced time series clustering capabilities.
    *   **Anomaly/Changepoint Detection:** State-of-the-art detection methods.
    *   **Scikit-learn Compatibility:** Integrates seamlessly with scikit-learn tools for model building, tuning, and validation.
    *   **Extensibility:** Easy to extend with custom algorithms.
*   📢 **Version 0.38.4 is now available!** [Read the Release Notes](https://www.sktime.net/en/latest/changelog.html).
*   📚 **Documentation:**

    *   [Tutorials](https://www.sktime.net/en/latest/tutorials.html): Get started with sktime.
    *   [Examples](https://www.sktime.net/en/latest/examples.html): Learn how to use sktime features.
    *   [API Reference](https://www.sktime.net/en/latest/api_reference.html): Detailed API reference.
    *   [Changelog](https://www.sktime.net/en/latest/changelog.html): View changes and version history.
    *   [Roadmap](https://www.sktime.net/en/latest/roadmap.html): sktime's development plan.
*   🤝 **Community & Support:**

    *   [Discord](https://discord.com/invite/54ACzaFsn7)
    *   [GitHub Discussions](https://github.com/sktime/sktime/discussions)
    *   [Stack Overflow](https://stackoverflow.com/questions/tagged/sktime)
*   🛠️ **Installation:**

    *   **Prerequisites:** macOS X, Linux, Windows 8.1+, Python 3.8 - 3.12 (64-bit).
    *   **Package Managers:** `pip`, `conda` (via `conda-forge`).

        ```bash
        pip install sktime
        # or with all extras
        pip install sktime[all_extras]
        conda install -c conda-forge sktime
        ```

*   🚀 **Quickstart Examples:**

    ```python
    # Forecasting
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

    # Time Series Classification
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

*   🤝 **Get Involved:** [Contribute](https://www.sktime.net/en/latest/get_involved/contributing.html)

    *   [Mentoring Program](https://github.com/sktime/mentoring)
    *   [Meetings](https://calendar.google.com/calendar/u/0/embed?src=sktime.toolbox@gmail.com&ctz=UTC)
    *   [Developer Guides](https://www.sktime.net/en/latest/developer_guide.html)
    *   [Enhancement Proposals](https://github.com/sktime/enhancement-proposals)
    *   [Contributors](https://github.com/sktime/sktime/blob/main/CONTRIBUTORS.md)
    *   [Roles](https://www.sktime.net/en/latest/about/team.html)
    *   [Donate](https://opencollective.com/sktime)

*   🏆 **Hall of Fame:**  [Contributors](https://github.com/sktime/sktime/graphs/contributors)
    <br>
    [![Contributors](https://opencollective.com/sktime/contributors.svg?width=600&button=false)](https://github.com/sktime/sktime/graphs/contributors)

*   💡 **Project Vision:**

    *   Community-driven development.
    *   The right tool for the right task.
    *   Integration with leading ecosystems.
    *   Model composition and feature engineering tools.
    *   Clear and descriptive syntax.
    *   Fair model assessment and benchmarking.
    *   Easy extensibility.