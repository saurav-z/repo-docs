<p align="center">
  <a target="_blank" href="https://transform.co/metricflow">
    <picture>
      <img  alt="MetricFlow Logo" src="https://github.com/dbt-labs/metricflow/raw/main/assets/MetricFlow_logo.png" width="auto" height="120">
    </picture>
  </a>
  <br /><br />
</p>

# MetricFlow: Build and Manage Metrics in Code

**MetricFlow simplifies your data analysis by allowing you to define and manage your metrics in code, ensuring consistency and accuracy.**

[![Twitter](https://img.shields.io/twitter/follow/dbt_labs?labelColor=image.png&color=163B36&logo=twitter&style=flat)](https://twitter.com/dbt_labs)
[![Slack](https://img.shields.io/badge/Slack-join-163B36)](https://www.getdbt.com/community/)
[![GitHub Stars](https://img.shields.io/github/stars/dbt-labs/metricflow?labelColor=image.png&color=163B36&logo=github)](https://github.com/dbt-labs/metricflow)
[![License](https://img.shields.io/pypi/l/metricflow?color=163B36&logo=AGPL-3.0)](https://github.com/dbt-labs/metricflow/blob/master/LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/metricflow?labelColor=&color=163B36)](https://pypi.org/project/metricflow/)
[![Python Versions](https://img.shields.io/pypi/pyversions/metricflow?labelColor=&color=163B36)](https://pypi.org/project/metricflow/)

Stay updated with the latest changes: [MetricFlow Changelog](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)

MetricFlow is a powerful semantic layer designed to streamline the definition and management of your key business metrics. By compiling metric definitions into reusable SQL, MetricFlow guarantees accurate and consistent results when analyzing your data across different dimensions.

<p align="center">
<img src="https://github.com/dbt-labs/metricflow/raw/main/assets/example_plan.svg" height="500"/>
<br /><br />
</p>

## Key Features

*   **Code-Based Metric Definitions:** Define and maintain all of your metric logic directly in code, promoting version control and reusability.
*   **Multi-Hop Joins:** Effortlessly handle complex joins between fact and dimension sources.
*   **Advanced Metric Types:** Support for a wide range of metric types, including ratio, expression, and cumulative metrics.
*   **Time Granularity Aggregation:** Aggregate your metrics to different time granularities for flexible analysis.
*   **Dataflow-Based Query Planning:** MetricFlow uses a dataflow-based approach to compile metric requests, optimizing and translating them into engine-specific SQL.

## Getting Started

### Installation

Install MetricFlow from PyPi for use as a Python library:

```bash
pip install dbt-metricflow
```

**Prerequisites:**

*   A working dbt project.
*   A dbt adapter.  The `dbt-metricflow` bundle is provided for this purpose.

You may need to install Postgres or Graphviz. You can do so by following the install instructions for [Postgres](https://www.postgresql.org/download/) or [Graphviz](https://www.graphviz.org/download/). Mac users may prefer to use brew: `brew install postgresql` or `brew install graphviz`.

### Tutorial

Get started with your own metrics by following the tutorial steps:

```bash
mf tutorial
```

*Note: Run this command from a dbt project root directory.*

## Resources

*   **Website:** [https://transform.co/metricflow](https://transform.co/metricflow)
*   **Documentation:** [https://docs.getdbt.com/docs/build/build-metrics-intro](https://docs.getdbt.com/docs/build/build-metrics-intro)
*   **Slack Community:** [https://www.getdbt.com/community/](https://www.getdbt.com/community/)
*   **GitHub Repository:** [https://github.com/dbt-labs/metricflow](https://github.com/dbt-labs/metricflow)
*   **CHANGELOG.md:** [https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)
*   **TENETS.md:** [https://github.com/dbt-labs/metricflow/blob/main/TENETS.md](https://github.com/dbt-labs/metricflow/blob/main/TENETS.md)

## Contributing

We welcome contributions! Please review our [code of conduct](https://docs.getdbt.com/community/resources/code-of-conduct) and [contributor guide](https://github.com/dbt-labs/metricflow/blob/main/CONTRIBUTING.md) to get started.

## Licensing

MetricFlow is source-available software, licensed under a Business Source License (BUSL-1.1).

MetricFlow is built by [dbt Labs](https://www.getdbt.com/).