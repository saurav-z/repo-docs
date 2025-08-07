# sktime: A Unified Toolkit for Time Series Analysis

[![sktime Logo](https://github.com/sktime/sktime/blob/main/docs/source/images/sktime-logo.svg?raw=true)](https://www.sktime.net)

**Analyze time series data with ease!** sktime is a comprehensive Python library providing a unified interface for various time series learning tasks, including forecasting, classification, clustering, and anomaly detection.  [Explore the sktime repository](https://github.com/sktime/sktime).

**Key Features:**

*   **Unified Interface:**  A consistent API for diverse time series tasks, simplifying model building and evaluation.
*   **Wide Range of Algorithms:**  Includes built-in time series algorithms and integration with popular libraries like scikit-learn.
*   **Model Composition:** Tools for building, tuning, and validating complex time series models using pipelines and ensembles.
*   **Task Support:** Forecasting, Classification, Regression, Clustering, Anomaly/Changepoint Detection, and more.
*   **Extensibility:** Easily extend sktime with your own algorithms using provided templates.

**Key Highlights:**

*   :rocket: **Version 0.38.4 is now available!**  [View Release Notes](https://www.sktime.net/en/latest/changelog.html)
*   :books: **Comprehensive Documentation:** Detailed guides, tutorials, examples, and API references.
*   :open_file_folder: **Open Source:**  Developed and maintained by a community, licensed under BSD-3.
*   :computer: **Active Community:** Engage with the community through [Discord](https://discord.com/invite/54ACzaFsn7) and [GitHub Discussions](https://github.com/sktime/sktime/discussions).

**Quick Links:**

*   :books: [Documentation](https://www.sktime.net/en/stable/users.html)
*   :rocket: [Release Notes](https://www.sktime.net/en/latest/changelog.html)
*   :star: [Tutorials](https://www.sktime.net/en/latest/tutorials.html)
*   :question: [Ask Questions](https://github.com/sktime/sktime/discussions)

## Core Features and Capabilities

sktime empowers you to tackle a wide array of time series challenges:

*   **Forecasting:**  Predict future values in time series data.
*   **Classification:** Categorize time series data into predefined classes.
*   **Regression:**  Predict continuous values based on time series data.
*   **Clustering:** Group similar time series together.
*   **Anomaly/Changepoint Detection:** Identify unusual patterns or significant shifts in time series.
*   **Transformations:** Apply various transformations to time series data for analysis and modeling.
*   **Time Series Distances/Kernels:** Measure similarity between time series.

**Modules & Status:**

| Module                      | Status     | Links                                                                                                                                  |
| --------------------------- | ---------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Forecasting                 | Stable     | [Tutorial](https://www.sktime.net/en/latest/examples/01_forecasting.html) · [API Reference](https://www.sktime.net/en/latest/api_reference/forecasting.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/forecasting.py) |
| Time Series Classification  | Stable     | [Tutorial](https://github.com/sktime/sktime/blob/main/examples/02_classification.ipynb) · [API Reference](https://www.sktime.net/en/latest/api_reference/classification.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/classification.py) |
| Time Series Regression      | Stable     | [API Reference](https://www.sktime.net/en/latest/api_reference/regression.html)                                                                                             |
| Transformations             | Stable     | [Tutorial](https://github.com/sktime/sktime/blob/main/examples/03_transformers.ipynb) · [API Reference](https://www.sktime.net/en/latest/api_reference/transformations.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/transformer.py) |
| Detection Tasks             | Maturing   | [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/detection.py)                                                                            |
| Parameter Fitting           | Maturing   | [API Reference](https://www.sktime.net/en/latest/api_reference/param_est.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/transformer.py) |
| Time Series Clustering      | Maturing   | [API Reference](https://www.sktime.net/en/latest/api_reference/clustering.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/clustering.py)         |
| Time Series Distances/Kernels | Maturing   | [Tutorial](https://github.com/sktime/sktime/blob/main/examples/03_transformers.ipynb) · [API Reference](https://www.sktime.net/en/latest/api_reference/dists_kernels.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/dist_kern_panel.py) |
| Time Series Alignment       | Experimental | [API Reference](https://www.sktime.net/en/latest/api_reference/alignment.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/alignment.py)        |
| Time Series Splitters       | Maturing   | [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/split.py)                                                                                    |
| Distributions & Simulation  | Experimental |  |

## Getting Started - Installation

Install sktime using pip or conda:

**Prerequisites:**

*   macOS X, Linux, or Windows 8.1 or higher
*   Python 3.8, 3.9, 3.10, 3.11, and 3.12 (64-bit only)

**pip:**

```bash
pip install sktime
```

Install with extra dependencies:

```bash
pip install sktime[all_extras]
```

For specific task dependencies, install with relevant extras:

```bash
pip install sktime[forecasting]
pip install sktime[forecasting,transformations]
```

Valid sets include: `forecasting`, `transformations`, `classification`, `regression`, `clustering`, `param_est`, `networks`, `detection`, and `alignment`.

**conda:**
```bash
conda install -c conda-forge sktime
```
Install with extra dependencies:
```bash
conda install -c conda-forge sktime-all-extras
```

## Quickstart Examples

Here are two simple examples to get you started:

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

## Join the sktime Community

Contribute and collaborate with the sktime community:

*   :bulb: **Contribute**: [How to contribute](https://www.sktime.net/en/latest/get_involved/contributing.html)
*   :school_satchel: **Mentoring**: [Apply to our mentoring program](https://github.com/sktime/mentoring)
*   :date: **Meetings**: [Join our discussions, tutorials, workshops, and sprints](https://calendar.google.com/calendar/u/0/embed?src=sktime.toolbox@gmail.com&ctz=UTC)
*   :woman_mechanic: **Developer Guides**: [How to develop sktime's code base](https://www.sktime.net/en/latest/developer_guide.html)
*   :construction: **Enhancement Proposals**: [Design a new feature](https://github.com/sktime/enhancement-proposals)
*   :medal_sports: **Contributors**: [List of all contributors](https://github.com/sktime/sktime/blob/main/CONTRIBUTORS.md)
*   :raising_hand: **Roles**: [Overview of our core community roles](https://www.sktime.net/en/latest/about/team.html)
*   :money_with_wings: **Donate**: [Fund sktime development](https://opencollective.com/sktime)
*   :classical_building: **Governance**: [How decisions are made](https://www.sktime.net/en/latest/get_involved/governance.html)

## Acknowledgements

<a href="https://github.com/sktime/sktime/graphs/contributors">
<img src="https://opencollective.com/sktime/contributors.svg?width=600&button=false" />
</a>
<br>

## Project Vision

*   **Community-Driven**: Developed by and for a friendly, collaborative community.
*   **Task-Oriented**: Helps users diagnose their learning problems and choose the right models.
*   **Interoperable**: Integrates seamlessly with scikit-learn, statsmodels, tsfresh, and other tools.
*   **Model Composition**: Offers functionality for building, tuning, and feature extraction pipelines.
*   **Clean Syntax**: Uses modern, object-oriented design principles.
*   **Fair Assessment**: Facilitates model inspection, validation, and avoidance of common pitfalls.
*   **Extensible**: Provides templates to easily add your own algorithms.