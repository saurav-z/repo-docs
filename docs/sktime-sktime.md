# sktime: Your Unified Toolkit for Time Series Analysis

<a href="https://www.sktime.net"><img src="https://github.com/sktime/sktime/blob/main/docs/source/images/sktime-logo.svg" width="175" align="right" /></a>

**sktime empowers you to easily build, evaluate, and deploy time series models for forecasting, classification, and more.** This Python library provides a unified interface for a wide range of time series learning tasks, offering a comprehensive toolkit for data scientists and researchers.  [**Visit the sktime repository on GitHub**](https://github.com/sktime/sktime)

**Key Features:**

*   **Unified Interface:** Provides a consistent and user-friendly API for diverse time series tasks.
*   **Comprehensive Task Coverage:** Supports forecasting, time series classification, regression, clustering, anomaly detection, and more.
*   **Extensive Algorithm Library:** Includes a rich collection of time series algorithms.
*   **scikit-learn Compatibility:** Offers tools for model building, tuning, and validation that are compatible with the scikit-learn ecosystem.
*   **Interoperability:** Seamlessly integrates with libraries like scikit-learn, statsmodels, tsfresh, and PyOD.
*   **Model Composition:** Enables building pipelines, ensembling, and tuning models for enhanced performance.
*   **Easy Extensibility:** Allows users to extend the library by adding custom algorithms and functionalities.
*   **Active Community:** Developed and maintained by a friendly and collaborative community.

**Version:** 0.38.4
[Check out the release notes here](https://www.sktime.net/en/latest/changelog.html).

## :books: Documentation & Resources

*   **[Documentation](https://www.sktime.net/en/stable/users.html)**: Comprehensive documentation for getting started and advanced use.
*   **[Tutorials](https://www.sktime.net/en/stable/examples.html)**: Step-by-step guides and examples to learn sktime.
*   **[Release Notes](https://www.sktime.net/en/stable/changelog.html)**: Stay updated on the latest changes and improvements.

**Additional Resources:**

*   **[Tutorials]**: New to sktime? Here's everything you need to know!
*   **[Binder Notebooks]**: Example notebooks to play with in your browser.
*   **[Examples]**: How to use sktime and its features.
*   **[Video Tutorial]**: Our video tutorial from 2021 PyData Global.
*   **[API Reference]**: The detailed reference for sktime's API.
*   **[Changelog]**: Changes and version history.
*   **[Roadmap]**: sktime's software and community development plan.
*   **[Related Software]**: A list of related software.

## :speech_balloon: Get Involved: Ask Questions and Contribute

*   **[GitHub Issue Tracker]**: Report bugs and suggest feature requests.
*   **[GitHub Discussions]**: Engage in general discussions and ask usage questions.
*   **[Stack Overflow]**: Find answers to your questions.
*   **[Discord]**: Join the community for discussions and development.

[github issue tracker]: https://github.com/sktime/sktime/issues
[github discussions]: https://github.com/sktime/sktime/discussions
[stack overflow]: https://stackoverflow.com/questions/tagged/sktime
[discord]: https://discord.com/invite/54ACzaFsn7

## :dizzy: Core Features & Modules

sktime provides a unified interface for distinct but related time series learning tasks. It features [__dedicated time series algorithms__](https://www.sktime.net/en/stable/estimator_overview.html) and __tools for composite model building__,  such as pipelining, ensembling, tuning, and reduction, empowering users to apply algorithms designed for one task to another.

sktime also provides **interfaces to related libraries**, for example [scikit-learn], [statsmodels], [tsfresh], [PyOD], and [fbprophet], among others.

[statsmodels]: https://www.statsmodels.org/stable/index.html
[tsfresh]: https://tsfresh.readthedocs.io/en/latest/
[pyod]: https://pyod.readthedocs.io/en/latest/
[fbprophet]: https://facebook.github.io/prophet/

| Module | Status | Links |
|---|---|---|
| **[Forecasting]** | stable | [Tutorial](https://www.sktime.net/en/latest/examples/01_forecasting.html) · [API Reference](https://www.sktime.net/en/latest/api_reference/forecasting.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/forecasting.py)  |
| **[Time Series Classification]** | stable | [Tutorial](https://github.com/sktime/sktime/blob/main/examples/02_classification.ipynb) · [API Reference](https://www.sktime.net/en/latest/api_reference/classification.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/classification.py) |
| **[Time Series Regression]** | stable | [API Reference](https://www.sktime.net/en/latest/api_reference/regression.html) |
| **[Transformations]** | stable | [Tutorial](https://github.com/sktime/sktime/blob/main/examples/03_transformers.ipynb) · [API Reference](https://www.sktime.net/en/latest/api_reference/transformations.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/transformer.py)  |
| **[Detection tasks]** | maturing | [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/detection.py) |
| **[Parameter fitting]** | maturing | [API Reference](https://www.sktime.net/en/latest/api_reference/param_est.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/transformer.py)  |
| **[Time Series Clustering]** | maturing | [API Reference](https://www.sktime.net/en/latest/api_reference/clustering.html) ·  [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/clustering.py) |
| **[Time Series Distances/Kernels]** | maturing | [Tutorial](https://github.com/sktime/sktime/blob/main/examples/03_transformers.ipynb) · [API Reference](https://www.sktime.net/en/latest/api_reference/dists_kernels.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/dist_kern_panel.py) |
| **[Time Series Alignment]** | experimental | [API Reference](https://www.sktime.net/en/latest/api_reference/alignment.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/alignment.py) |
| **[Time Series Splitters]** | maturing | [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/split.py) | |
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

## :hourglass_flowing_sand: Installation

See the [documentation](https://www.sktime.net/en/latest/installation.html) for detailed installation instructions.

*   **Operating System:** macOS X, Linux, Windows 8.1 or higher
*   **Python Version:** 3.8, 3.9, 3.10, 3.11, and 3.12 (64-bit only)
*   **Package Managers:** [pip] and [conda] (via `conda-forge`)

### pip

```bash
pip install sktime
```

Install with all extras:

```bash
pip install sktime[all_extras]
```

Curated sets of soft dependencies for specific learning tasks:

```bash
pip install sktime[forecasting]  # for selected forecasting dependencies
pip install sktime[forecasting,transformations]  # forecasters and transformers
```

or similar. Valid sets are:

*   `forecasting`
*   `transformations`
*   `classification`
*   `regression`
*   `clustering`
*   `param_est`
*   `networks`
*   `detection`
*   `alignment`

Cave: in general, not all soft dependencies for a learning task are installed,
only a curated selection.

### conda

```bash
conda install -c conda-forge sktime
```

Install with all extras:

```bash
conda install -c conda-forge sktime-all-extras
```

## :zap: Quickstart Examples

### Forecasting

``` python
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

## :wave: How to Get Involved

Join the sktime community!

*   **[Contribute]**: Learn how to contribute to sktime.
*   **[Mentoring]**: Apply to our mentoring program.
*   **[Meetings]**: Join our discussions, tutorials, workshops, and sprints.
*   **[Developer Guides]**: Develop sktime's code base.
*   **[Enhancement Proposals]**: Design new features.
*   **[Contributors]**: See a list of all contributors.
*   **[Roles]**: Review our core community roles.
*   **[Donate]**: Support sktime maintenance and development.
*   **[Governance]**: Understand how decisions are made in the community.

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

## :trophy: Hall of Fame

Thanks to all our community for all your wonderful contributions, PRs, issues, ideas.

<a href="https://github.com/sktime/sktime/graphs/contributors">
<img src="https://opencollective.com/sktime/contributors.svg?width=600&button=false" />
</a>
<br>

## :bulb: Project Vision

*   **By the community, for the community**: Developed by a friendly and collaborative community.
*   The **right tool for the right task**: Helping users diagnose learning problems and choose suitable models.
*   **Embedded in state-of-art ecosystems** and **provider of interoperable interfaces**: Interoperable with scikit-learn, statsmodels, tsfresh, and other community favorites.
*   **Rich model composition and reduction functionality**: Build tuning and feature extraction pipelines, solve forecasting tasks with scikit-learn regressors.
*   **Clean, descriptive specification syntax**: Based on modern object-oriented design principles for data science.
*   **Fair model assessment and benchmarking**: Build, inspect, and check models to avoid pitfalls.
*   **Easily extensible**: Easy extension templates to add your own algorithms compatible with sktime's API.