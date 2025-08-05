# sktime: A Unified Python Library for Time Series Analysis

> **sktime simplifies time series analysis, offering a consistent interface for forecasting, classification, and more.** Explore the latest release: [sktime v0.38.4](https://www.sktime.net/en/latest/changelog.html).

[![sktime Logo](https://github.com/sktime/sktime/blob/main/docs/source/images/sktime-logo.svg?raw=true)](https://www.sktime.net/)

**Key Features:**

*   **Unified Interface:** Provides a consistent API for various time series tasks, including forecasting, classification, clustering, and anomaly detection.
*   **Comprehensive Algorithms:**  Includes a wide range of time series algorithms.
*   **Scikit-learn Compatibility:** Integrates seamlessly with scikit-learn tools for model building, tuning, and validation.
*   **Extensive Documentation:** Detailed documentation, tutorials, and examples to get you started quickly.
*   **Community-Driven:**  Actively maintained and supported by a vibrant and welcoming community.
*   **Open Source:**  Available under the BSD 3-Clause License.

[**Explore the sktime documentation!**](https://www.sktime.net/en/stable/users.html)

## Core Functionality

sktime offers a robust suite of tools and functionalities to tackle various time series challenges:

*   **Forecasting:** Build and evaluate forecasting models.
*   **Time Series Classification:**  Classify time series data.
*   **Time Series Regression:** Perform regression tasks on time series.
*   **Transformations:** Apply various data transformations.
*   **Clustering:**  Group similar time series together.
*   **Anomaly Detection:** Identify unusual patterns in time series data.

## Resources

### Documentation

*   :star: **[Tutorials](https://www.sktime.net/en/latest/tutorials.html)**: New to sktime? Here's everything you need to know!
*   :clipboard: **[Binder Notebooks](https://mybinder.org/v2/gh/sktime/sktime/main?filepath=examples)**: Interactive examples to try in your browser.
*   :woman_technologist: **[Examples](https://www.sktime.net/en/latest/examples.html)**: Practical examples showcasing sktime's features.
*   :scissors: **[Extension Templates](https://github.com/sktime/sktime/blob/main/extension_templates)**: Build your own estimators.
*   :control_knobs: **[API Reference](https://www.sktime.net/en/latest/api_reference.html)**: Detailed API documentation.
*   :tv: **[Video Tutorial](https://github.com/sktime/sktime-tutorial-pydata-global-2021)**: Video tutorial from PyData Global 2021.
*   :hammer_and_wrench: **[Changelog](https://www.sktime.net/en/latest/changelog.html)**: Changes and version history.
*   :deciduous_tree: **[Roadmap](https://www.sktime.net/en/latest/roadmap.html)**: sktime's development plan.
*   :pencil: **[Related Software](https://www.sktime.net/en/latest/related_software.html)**: A list of related software.

### Community & Support

*   :bug: **Bug Reports & Feature Requests:** [GitHub Issue Tracker](https://github.com/sktime/sktime/issues)
*   :speech_balloon: **Discussions:** [GitHub Discussions](https://github.com/sktime/sktime/discussions)
*   :woman_technologist: **Usage Questions:** [Stack Overflow](https://stackoverflow.com/questions/tagged/sktime)
*   :factory: **Contribution & Development:** `dev-chat` channel and [Discord](https://discord.com/invite/54ACzaFsn7)
*   :globe_with_meridians: **Meet-ups & Collaboration:** [Discord](https://discord.com/invite/54ACzaFsn7) - Fridays 13 UTC, dev/meet-ups channel

## Installation

Install sktime using pip or conda:

**Pip:**

```bash
pip install sktime
```

**Conda:**

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

Contribute to sktime and become part of the community:

*   :gift_heart: **[Contribute](https://www.sktime.net/en/latest/get_involved/contributing.html)**: How to contribute.
*   :school_satchel: **[Mentoring](https://github.com/sktime/mentoring)**: Apply to our mentoring program.
*   :date: **[Meetings](https://calendar.google.com/calendar/u/0/embed?src=sktime.toolbox@gmail.com&ctz=UTC)**: Join our discussions, tutorials, and sprints!
*   :woman_mechanic: **[Developer Guides](https://www.sktime.net/en/latest/developer_guide.html)**: Develop sktime's code base.
*   :construction: **[Enhancement Proposals](https://github.com/sktime/enhancement-proposals)**: Design new features.
*   :medal_sports: **[Contributors](https://github.com/sktime/sktime/blob/main/CONTRIBUTORS.md)**: List of all contributors.
*   :raising_hand: **[Roles](https://www.sktime.net/en/latest/about/team.html)**: Core community roles.
*   :money_with_wings: **[Donate](https://opencollective.com/sktime)**: Fund sktime development.
*   :classical_building: **[Governance](https://www.sktime.net/en/latest/get_involved/governance.html)**: How decisions are made.

## Hall of Fame

A special thanks to all our amazing contributors!

[![Contributors](https://opencollective.com/sktime/contributors.svg?width=600&button=false)](https://github.com/sktime/sktime/graphs/contributors)

## Project Vision

*   **Community-Driven:** Developed by a collaborative community.
*   **Task-Specific Tools:** Provides the right tools for each task.
*   **Ecosystem Integration:** Interoperable with key libraries.
*   **Model Composition:** Rich functionality for model building and feature engineering.
*   **Clean Design:** Based on modern object-oriented design principles.
*   **Fair Assessment:** Emphasizes model inspection and validation.
*   **Extensible:** Easy to add your own algorithms.

[**Return to the sktime repository!**](https://github.com/sktime/sktime)