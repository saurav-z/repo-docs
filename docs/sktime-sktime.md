<!-- Improved README - SEO Optimized -->
# sktime: Unified Machine Learning for Time Series Analysis

[![sktime Logo](https://github.com/sktime/sktime/blob/main/docs/source/images/sktime-logo.svg?raw=true)](https://www.sktime.net)

> **sktime empowers you to easily build, evaluate, and deploy time series models with a scikit-learn-like API!**

**[Visit the sktime GitHub Repository](https://github.com/sktime/sktime)**

:rocket: **Version 0.38.4 is available!** [See the release notes](https://www.sktime.net/en/latest/changelog.html).

sktime is a comprehensive Python library designed for time series analysis, offering a unified interface for various machine learning tasks.  It provides a consistent and intuitive approach to time series forecasting, classification, clustering, anomaly detection, and more.  With a vast collection of time series algorithms and tools compatible with [scikit-learn], sktime simplifies the process of building, tuning, and validating your time series models.

**Key Features:**

*   **Unified Interface:**  A consistent API for diverse time series tasks.
*   **Extensive Algorithms:**  Includes a wide range of forecasting and time series learning algorithms.
*   **scikit-learn Compatibility:** Integrates seamlessly with [scikit-learn] tools for building, tuning, and validating models.
*   **Task-Specific Functionality:** Supports forecasting, classification, clustering, anomaly detection, and other time series analysis tasks.
*   **Model Composition:** Features tools for pipelining, ensembling, and model selection.
*   **Integration with other Libraries:** Provides interfaces to related libraries, including [statsmodels], [tsfresh], [PyOD], and [fbprophet].
*   **Extensible Design:**  Easy-to-use extension templates for incorporating custom algorithms.

**Quick Links:**

| Resource                         | Link                                                                     |
| -------------------------------- | ------------------------------------------------------------------------ |
| **Documentation**               | [Documentation](https://www.sktime.net/en/stable/users.html)           |
| **Tutorials**                   | [Tutorials](https://www.sktime.net/en/latest/tutorials.html)           |
| **Release Notes**               | [Release Notes](https://www.sktime.net/en/latest/changelog.html)           |
| **Estimator Overview**          | [Time Series Algorithms](https://www.sktime.net/en/stable/estimator_overview.html)    |
| **Community Forum**             | [Discussions](https://github.com/sktime/sktime/discussions)            |
| **GitHub Repository**             | [GitHub](https://github.com/sktime/sktime)          |
| **Open Source License**         | [BSD 3-clause](https://github.com/sktime/sktime/blob/main/LICENSE) |

<br>

### :books: Documentation and Tutorials

Explore the detailed documentation and examples to get started:

*   **:star: [Tutorials]**:  A great starting point for understanding sktime.
*   **:clipboard: [Binder Notebooks]**: Interactive examples in your browser.
*   **:woman_technologist: [Examples]**:  Practical guides on using sktime.
*   **:scissors: [Extension Templates]**: Build your own estimators using sktime's API.
*   **:control_knobs: [API Reference]**:  Detailed API documentation.
*   **:tv: [Video Tutorial]**:  PyData Global 2021 tutorial.
*   **:hammer_and_wrench: [Changelog]**:  View changes and version history.
*   **:deciduous_tree: [Roadmap]**:  sktime's development plan.
*   **:pencil: [Related Software]**:  Related software integrations.

[tutorials]: https://www.sktime.net/en/latest/tutorials.html
[binder notebooks]: https://mybinder.org/v2/gh/sktime/sktime/main?filepath=examples
[examples]: https://www.sktime.net/en/latest/examples.html
[video tutorial]: https://github.com/sktime/sktime-tutorial-pydata-global-2021
[api reference]: https://www.sktime.net/en/latest/api_reference.html
[changelog]: https://www.sktime.net/en/latest/changelog.html
[roadmap]: https://www.sktime.net/en/latest/roadmap.html
[related software]: https://www.sktime.net/en/latest/related_software.html

<br>

### :speech_balloon: Get Support and Contribute

We highly value community contributions.  Find help and connect with other users:

| Type                            | Platforms                               |
| ------------------------------- | --------------------------------------- |
| :bug: **Bug Reports**              | [GitHub Issue Tracker]                  |
| :sparkles: **Feature Requests & Ideas** | [GitHub Issue Tracker]                       |
| :woman_technologist: **Usage Questions**          | [GitHub Discussions] · [Stack Overflow] |
| :speech_balloon: **General Discussion**        | [GitHub Discussions] |
| :factory: **Contribution & Development** | `dev-chat` channel · [Discord] |
| :globe_with_meridians: **Meet-ups and collaboration sessions** | [Discord] - Fridays 13 UTC, dev/meet-ups channel |

[github issue tracker]: https://github.com/sktime/sktime/issues
[github discussions]: https://github.com/sktime/sktime/discussions
[stack overflow]: https://stackoverflow.com/questions/tagged/sktime
[discord]: https://discord.com/invite/54ACzaFsn7

<br>

### :dizzy: Available Tasks

sktime offers a range of functionalities for time series analysis:

*   **Forecasting**
*   **Time Series Classification**
*   **Time Series Regression**
*   **Transformations**
*   **Detection tasks**
*   **Parameter fitting**
*   **Time Series Clustering**
*   **Time Series Distances/Kernels**
*   **Time Series Alignment**
*   **Time Series Splitters**
*   **Distributions and simulation**

**Note:**  See the full range of functionality in the table below, including links to specific modules.

| Module                            | Status   | Links                                                                                             |
| --------------------------------- | -------- | ------------------------------------------------------------------------------------------------- |
| **[Forecasting]**                | stable   | [Tutorial](https://www.sktime.net/en/latest/examples/01_forecasting.html)  · [API Reference](https://www.sktime.net/en/latest/api_reference/forecasting.html)  · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/forecasting.py)  |
| **[Time Series Classification]**  | stable   | [Tutorial](https://github.com/sktime/sktime/blob/main/examples/02_classification.ipynb) · [API Reference](https://www.sktime.net/en/latest/api_reference/classification.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/classification.py) |
| **[Time Series Regression]**      | stable   | [API Reference](https://www.sktime.net/en/latest/api_reference/regression.html) |
| **[Transformations]**             | stable   | [Tutorial](https://github.com/sktime/sktime/blob/main/examples/03_transformers.ipynb) · [API Reference](https://www.sktime.net/en/latest/api_reference/transformations.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/transformer.py)  |
| **[Detection tasks]**             | maturing | [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/detection.py) |
| **[Parameter fitting]**           | maturing | [API Reference](https://www.sktime.net/en/latest/api_reference/param_est.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/transformer.py)  |
| **[Time Series Clustering]**     | maturing | [API Reference](https://www.sktime.net/en/latest/api_reference/clustering.html) ·  [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/clustering.py) |
| **[Time Series Distances/Kernels]** | maturing | [Tutorial](https://github.com/sktime/sktime/blob/main/examples/03_transformers.ipynb) · [API Reference](https://www.sktime.net/en/latest/api_reference/dists_kernels.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/dist_kern_panel.py) |
| **[Time Series Alignment]**       | experimental | [API Reference](https://www.sktime.net/en/latest/api_reference/alignment.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/alignment.py) |
| **[Time Series Splitters]**       | maturing | [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/split.py) |
| **[Distributions and simulation]** | experimental |  |

[forecasting]: https://github.com/sktime/sktime/tree/main/sktime/forecasting
[time series classification]: https://github.com/sktime/sktime/tree/main/sktime/classification
[time series regression]: https://github.com/sktime/sktime/tree/main/sktime/regression
[time series clustering]: https://github.com/sktime/sktime/tree/main/sktime/clustering
[detection tasks]: https://github.com/sktime/sktime/tree/main/sktime/detection
[time series distances/kernels]: https://github.com/sktime/sktime/tree/main/sktime/dists_kernels
[time series alignment]: https://github.com/sktime/sktime/tree/main/sktime/alignment
[transformations]: https://github.com/sktime/sktime/tree/main/sktime/transformations
[distributions and simulation]: https://github.com/sktime/sktime/tree/main/sktime/proba
[time series splitters]: https://github.com/sktime/sktime/tree/main/sktime/split
[parameter fitting]: https://github.com/sktime/sktime/tree/main/sktime/param_est

<br>

### :hourglass_flowing_sand: Installation

Easily install sktime using pip or conda.

**System Requirements:**

*   **Operating System:** macOS X, Linux, or Windows 8.1+
*   **Python Version:** 3.8, 3.9, 3.10, 3.11, and 3.12 (64-bit only)

**Installation Options:**

*   **pip:**  `pip install sktime` or  `pip install sktime[all_extras]` for all dependencies.  Specify task-specific dependencies like  `pip install sktime[forecasting]` or `pip install sktime[forecasting,transformations]`.
*   **conda:**  `conda install -c conda-forge sktime`  or `conda install -c conda-forge sktime-all-extras` (for all dependencies).

<br>

### :zap: Quickstart Examples

Get started quickly with these example code snippets.

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

<br>

### :wave: Get Involved

Become part of the sktime community! We welcome contributions of all types.

| Documentation              |                                                                |
| -------------------------- | --------------------------------------------------------------        |
| :gift_heart: **[Contribute]**        | How to contribute to sktime.          |
| :school_satchel:  **[Mentoring]** | New to open source? Apply to our mentoring program! |
| :date: **[Meetings]** | Join our discussions, tutorials, workshops, and sprints! |
| :woman_mechanic:  **[Developer Guides]**      | How to further develop sktime's code base.                             |
| :construction: **[Enhancement Proposals]** | Design a new feature for sktime. |
| :medal_sports: **[Contributors]** | A list of all contributors. |
| :raising_hand: **[Roles]** | An overview of our core community roles. |
| :money_with_wings: **[Donate]** | Fund sktime maintenance and development. |
| :classical_building: **[Governance]** | How and by whom decisions are made in sktime's community.   |

[contribute]: https://www.sktime.net/en/latest/get_involved/contributing.html
[donate]: https://opencollective.com/sktime
[extension templates]: https://github.com/sktime/sktime/tree/main/extension_templates
[developer guides]: https://www.sktime.net/en/latest/developer_guide.html
[contributors]: https://github.com/sktime/sktime/blob/main/CONTRIBUTORS.md
[governance]: https://www.sktime.net/en/latest/get_involved/governance.html
[mentoring]: https://github.com/sktime/mentoring
[meetings]: https://calendar.google.com/calendar/u/0/embed?src=sktime.toolbox@gmail.com&ctz=UTC
[enhancement proposals]: https://github.com/sktime/enhancement-proposals
[roles]: https://www.sktime.net/en/latest/about/team.html

<br>

### :trophy:  Hall of Fame

We are grateful to all our community members for their valuable contributions.

<a href="https://github.com/sktime/sktime/graphs/contributors">
<img src="https://opencollective.com/sktime/contributors.svg?width=600&button=false" />
</a>

<br>

### :bulb: Project Vision

*   **By the community, for the community:** A project developed by a friendly and collaborative community.
*   **The right tool for the right task:** Guiding users toward appropriate models.
*   **Embedded in state-of-the-art ecosystems:**  Interoperable with key libraries like [scikit-learn], [statsmodels], [tsfresh], and others.
*   **Rich model composition and reduction functionality:**  Supports building pipelines, tuning, and feature extraction.
*   **Clean, descriptive specification syntax:** Based on modern object-oriented design.
*   **Fair model assessment and benchmarking:**  Helping users to build, inspect, and evaluate models effectively.
*   **Easily extensible:**  Simple extension templates for integrating custom algorithms.