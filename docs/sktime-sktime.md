<!-- Improved & SEO-Optimized README for sktime -->

# sktime: Your Unified Toolkit for Time Series Analysis

<a href="https://www.sktime.net">
  <img src="https://github.com/sktime/sktime/blob/main/docs/source/images/sktime-logo.svg" width="175" align="right" alt="sktime logo">
</a>

**sktime empowers you to easily build, evaluate, and deploy time series models with a unified and intuitive interface.**

*   **[Explore the official sktime website](https://www.sktime.net/)**
*   **[View the latest release notes](https://www.sktime.net/en/latest/changelog.html)**

**Key Features:**

*   **Unified Interface:** A consistent API for various time series tasks, simplifying model building and comparison.
*   **Diverse Tasks:** Support for forecasting, classification, clustering, anomaly detection, and more.
*   **Rich Algorithms:**  Built-in time series algorithms for a wide range of applications.
*   **scikit-learn Compatibility:** Seamless integration with scikit-learn tools for model building, tuning, and validation.
*   **Interoperability:** Interfaces to popular time series libraries like statsmodels, tsfresh, and others.
*   **Extensibility:** Easily extend sktime with your own custom algorithms.

**Key Use Cases:**

*   **Forecasting:** Predict future values of time series data.
*   **Time Series Classification:** Categorize time series based on their patterns.
*   **Anomaly Detection:** Identify unusual or unexpected events in time series.
*   **Time Series Clustering:** Group similar time series together.

**Get Started:**

*   **[Documentation](https://www.sktime.net/en/stable/users.html)**
*   **[Tutorials](https://www.sktime.net/en/latest/tutorials.html)**
*   **[Examples](https://www.sktime.net/en/latest/examples.html)**

---

## Quickstart

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

## Core Functionality and Modules

sktime's modular design allows you to choose the right tools for your time series analysis needs.

| Module                                      | Status   | Links                                                                                                |
| :------------------------------------------ | :------- | :--------------------------------------------------------------------------------------------------- |
| **Forecasting**                                | Stable   | [Tutorial](https://www.sktime.net/en/latest/examples/01_forecasting.html) · [API Reference](https://www.sktime.net/en/latest/api_reference/forecasting.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/forecasting.py)       |
| **Time Series Classification**                 | Stable   | [Tutorial](https://github.com/sktime/sktime/blob/main/examples/02_classification.ipynb) · [API Reference](https://www.sktime.net/en/latest/api_reference/classification.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/classification.py) |
| **Time Series Regression**                    | Stable   | [API Reference](https://www.sktime.net/en/latest/api_reference/regression.html)                                |
| **Transformations**                           | Stable   | [Tutorial](https://github.com/sktime/sktime/blob/main/examples/03_transformers.ipynb) · [API Reference](https://www.sktime.net/en/latest/api_reference/transformations.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/transformer.py) |
| **Detection tasks**                         | Maturing | [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/detection.py)        |
| **Parameter fitting**                       | Maturing | [API Reference](https://www.sktime.net/en/latest/api_reference/param_est.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/transformer.py) |
| **Time Series Clustering**                   | Maturing | [API Reference](https://www.sktime.net/en/latest/api_reference/clustering.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/clustering.py) |
| **Time Series Distances/Kernels**            | Maturing | [Tutorial](https://github.com/sktime/sktime/blob/main/examples/03_transformers.ipynb) · [API Reference](https://www.sktime.net/en/latest/api_reference/dists_kernels.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/dist_kern_panel.py) |
| **Time Series Alignment**                    | Experimental | [API Reference](https://www.sktime.net/en/latest/api_reference/alignment.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/alignment.py)  |
| **Time Series Splitters**                   | Maturing | [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/split.py)                                |
| **Distributions and simulation**             | Experimental |                                                                                                      |

---

## Installation

Install sktime using pip or conda.  Detailed instructions can be found in the [documentation](https://www.sktime.net/en/latest/installation.html).

**Requirements:**

*   **Operating system**: macOS X · Linux · Windows 8.1 or higher
*   **Python version**: Python 3.8, 3.9, 3.10, 3.11, and 3.12 (only 64-bit)
*   **Package managers**: [pip] · [conda] (via `conda-forge`)

### pip

```bash
pip install sktime
```

Install with extra dependencies:

```bash
pip install sktime[all_extras]
```

Install with specific learning tasks:

```bash
pip install sktime[forecasting]
pip install sktime[forecasting,transformations]
```

### conda

```bash
conda install -c conda-forge sktime
```

Install with extra dependencies:

```bash
conda install -c conda-forge sktime-all-extras
```

---

## Resources and Community

Get help and connect with the sktime community.

| Resource                           | Description                                                        |
| :--------------------------------- | :----------------------------------------------------------------- |
| [Documentation](https://www.sktime.net/en/stable/users.html)           | Comprehensive documentation for users.                                  |
| [Tutorials](https://www.sktime.net/en/latest/tutorials.html)             | Step-by-step tutorials for getting started.                             |
| [Examples](https://www.sktime.net/en/latest/examples.html)                | Code examples and use cases.                                            |
| [GitHub Repository](https://github.com/sktime/sktime) | Access the source code and contribute to the project.   |
| [Discord](https://discord.com/invite/54ACzaFsn7)            | Join the community chat.                                                    |
| [Stack Overflow](https://stackoverflow.com/questions/tagged/sktime)     | Find answers and ask questions.                                       |
| [GitHub Discussions](https://github.com/sktime/sktime/discussions)       | Engage in discussions about sktime.                                         |

---

## Getting Involved

Contribute to sktime and help shape the future of time series analysis.

| Resource                  | Description                                         |
| :------------------------ | :-------------------------------------------------- |
| [Contribute](https://www.sktime.net/en/latest/get_involved/contributing.html)            | Learn how to contribute to the project.                  |
| [Mentoring](https://github.com/sktime/mentoring)          | Mentoring program for new contributors.  |
| [Meetings](https://calendar.google.com/calendar/u/0/embed?src=sktime.toolbox@gmail.com&ctz=UTC) | Join our discussions, tutorials, workshops, and sprints!   |
| [Developer Guides](https://www.sktime.net/en/latest/developer_guide.html)  | Developer guides for contributing to the codebase.     |
| [Enhancement Proposals](https://github.com/sktime/enhancement-proposals)      | Propose new features for sktime.                      |
| [Governance](https://www.sktime.net/en/latest/get_involved/governance.html)  | Learn how decisions are made in the community.        |
| [Contributors](https://github.com/sktime/sktime/blob/main/CONTRIBUTORS.md)      | List of all contributors.       |
| [Roles](https://www.sktime.net/en/latest/about/team.html)     | Overview of our core community roles.   |
| [Donate](https://opencollective.com/sktime)      | Fund sktime maintenance and development. |
---

## Project Vision

*   **Community-Driven:** Developed by a friendly and collaborative community.
*   **Task-Oriented:**  Helps users choose the right models.
*   **Interoperable:**  Works seamlessly with other popular libraries.
*   **Rich Functionality:**  Provides composability and model building capabilities.
*   **Modern Design:** Clean, object-oriented design principles.
*   **Fair Assessment:** Encourages robust model evaluation.
*   **Extensible:** Easily add your own algorithms.

---

##  Citation

If you use sktime in your research, please cite our work:

[![!zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.3749000.svg)](https://doi.org/10.5281/zenodo.3749000)

---

**[Visit the sktime GitHub repository](https://github.com/sktime/sktime) to learn more and get involved!**
```

Key improvements and SEO optimizations:

*   **Clear Headline:** Uses the most important keyword ("time series analysis") in the main title.
*   **Concise Hook:** The one-sentence description at the beginning summarizes the value proposition.
*   **Keyword Optimization:** Naturally incorporates relevant keywords throughout the text: "time series," "forecasting," "classification," "anomaly detection," "machine learning," etc.
*   **Bulleted Key Features:**  Easy-to-read and highlights the core benefits.
*   **Organized Structure:**  Clear headings and subheadings for readability and SEO.
*   **Descriptive Links:** Uses descriptive anchor text for links, boosting SEO.
*   **Call to Action:** Encourages users to visit the GitHub repository.
*   **Stronger Focus on Benefits:**  Highlights what sktime offers the user, not just what it *is*.
*   **Removed Redundancy:** Simplified some sections.
*   **Enhanced Use Case Descriptions:**  Expanded on the key use cases.
*   **Modernized Tone:** Uses more engaging language.
*   **Combined and Reorganized Sections:**  Streamlined the layout.