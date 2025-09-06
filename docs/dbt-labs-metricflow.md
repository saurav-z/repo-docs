<p align="center">
  <a target="_blank" href="https://transform.co/metricflow">
    <picture>
      <img  alt="MetricFlow logo" src="https://github.com/dbt-labs/metricflow/raw/main/assets/MetricFlow_logo.png" width="auto" height="120">
    </picture>
  </a>
  <br /><br />
</p>

# MetricFlow: Build Consistent Metrics in Code

**MetricFlow empowers data teams to define, manage, and consistently calculate metrics, enabling reliable and efficient data analysis.**

<br/>

[![Twitter](https://img.shields.io/twitter/follow/dbt_labs?labelColor=image.png&color=163B36&logo=twitter&style=flat)](https://twitter.com/dbt_labs)
[![Slack](https://img.shields.io/badge/Slack-join-163B36)](https://www.getdbt.com/community/)
[![GitHub stars](https://img.shields.io/github/stars/dbt-labs/metricflow?labelColor=image.png&color=163B36&logo=github)](https://github.com/dbt-labs/metricflow)
[![License](https://img.shields.io/pypi/l/metricflow?color=163B36&logo=AGPL-3.0)](https://github.com/dbt-labs/metricflow/blob/master/LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/metricflow?labelColor=&color=163B36)](https://pypi.org/project/metricflow/)
[![Python versions](https://img.shields.io/pypi/pyversions/metricflow?labelColor=&color=163B36)](https://pypi.org/project/metricflow/)

<br/>

**Key Features:**

*   **Centralized Metric Definitions:** Define metrics in code for version control, reusability, and collaboration.
*   **Dataflow-Based Query Optimization:** Compiles metric requests into efficient, optimized SQL queries.
*   **Multi-Hop Join Support:**  Handles complex joins between fact and dimension tables.
*   **Advanced Metric Types:** Supports ratio, expression, cumulative, and other advanced metric calculations.
*   **Time Granularity Aggregation:** Aggregates metrics to different time granularities for flexible analysis.
*   **Consistent Results:** Ensures accuracy and consistency in metric calculations across your data analysis.

<p align="center">
<img src="https://github.com/dbt-labs/metricflow/raw/main/assets/example_plan.svg" height="500"/>
<br /><br />
</p>

## Getting Started

### Installation

Install MetricFlow using pip:

```bash
pip install dbt-metricflow
```

MetricFlow integrates with dbt projects.  You will need a working dbt project and a dbt adapter. The `dbt-metricflow` package provides the necessary integration. You may optionally install other adapters as extras.

You may also need to install Postgres or Graphviz.  See the [Postgres](https://www.postgresql.org/download/) or [Graphviz](https://www.graphviz.org/download/) installation instructions.  Mac users can use `brew install postgresql` or `brew install graphviz`.

### Tutorial

Start learning with a step-by-step tutorial (run from a dbt project root directory):

```bash
mf tutorial
```

## Resources

*   **[MetricFlow Website](https://transform.co/metricflow)**
*   **[Documentation](https://docs.getdbt.com/docs/build/build-metrics-intro)**
*   **[Slack Community](https://www.getdbt.com/community/)**
*   **[MetricFlow GitHub Repository](https://github.com/dbt-labs/metricflow)**
*   [CHANGELOG.md](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)
*   [TENETS.md](https://github.com/dbt-labs/metricflow/blob/main/TENETS.md)

## Contributing

We welcome contributions! Please review our [code of conduct](https://docs.getdbt.com/community/resources/code-of-conduct) and [contributor guide](https://github.com/dbt-labs/metricflow/blob/main/CONTRIBUTING.md) to get started.

## Licensing

MetricFlow is source-available software.

*   Version 0 to 0.140.0: Affero GPL license.
*   Version 0.150.0 and greater: BSL license.

MetricFlow is built by [dbt Labs](https://www.getdbt.com/).