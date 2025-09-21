<p align="center">
  <a target="_blank" href="https://transform.co/metricflow">
    <picture>
      <img  alt="metricflow logo" src="https://github.com/dbt-labs/metricflow/raw/main/assets/MetricFlow_logo.png" width="auto" height="120">
    </picture>
  </a>
</p>

[![Twitter](https://img.shields.io/twitter/follow/dbt_labs?labelColor=image.png&color=163B36&logo=twitter&style=flat)](https://twitter.com/dbt_labs)
[![Slack](https://img.shields.io/badge/Slack-join-163B36)](https://www.getdbt.com/community/)
[![GitHub stars](https://img.shields.io/github/stars/dbt-labs/metricflow?labelColor=image.png&color=163B36&logo=github)](https://github.com/dbt-labs/metricflow)
[![License](https://img.shields.io/pypi/l/metricflow?color=163B36&logo=AGPL-3.0)](https://github.com/dbt-labs/metricflow/blob/master/LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/metricflow?labelColor=&color=163B36)](https://pypi.org/project/metricflow/)
[![Python versions](https://img.shields.io/pypi/pyversions/metricflow?labelColor=&color=163B36)](https://pypi.org/project/metricflow/)

# MetricFlow: Build and Manage Your Metrics in Code

**MetricFlow is a powerful semantic layer that empowers data teams to define, manage, and consistently apply key business metrics.**

[Check out the latest updates in the MetricFlow Changelog](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)!

MetricFlow streamlines metric definition, compiling your definitions into reusable SQL code, ensuring consistent and accurate data analysis across all your projects. This semantic layer handles the complexities of your data, from multi-hop joins to complex metric types, allowing you to focus on the meaning of your data.

<p align="center">
<img src="https://github.com/dbt-labs/metricflow/raw/main/assets/example_plan.svg" height="500"/>
<br /><br />
</p>

## Key Features of MetricFlow:

*   **Code-Based Metric Definitions:** Define your metrics in code for version control, reusability, and collaboration.
*   **Multi-Hop Joins:** Simplify complex joins between fact and dimension tables.
*   **Complex Metric Types:** Supports various metric types, including ratio, expression, and cumulative metrics.
*   **Time Granularity Aggregation:** Aggregate metrics to different time granularities for flexible analysis.
*   **Consistent Results:** Ensures consistent and accurate metric calculations across all analyses.

## Getting Started

### Installation

Install MetricFlow using pip:

```bash
pip install dbt-metricflow
```

MetricFlow is designed to work with a dbt project and adapter.

You may need to install Postgres or Graphviz. You can do so by following the install instructions for [Postgres](https://www.postgresql.org/download/) or [Graphviz](https://www.graphviz.org/download/). Mac users may prefer to use brew: `brew install postgresql` or `brew install graphviz`.

### Tutorial

Get started quickly with the tutorial:

```bash
mf tutorial
```

*Note: Run this command from a dbt project root directory.*

## Resources

*   [Website](https://transform.co/metricflow)
*   [Documentation](https://docs.getdbt.com/docs/build/build-metrics-intro)
*   [Slack Community](https://www.getdbt.com/community/)
*   [MetricFlow GitHub Repository](https://github.com/dbt-labs/metricflow)
*   [CHANGELOG.md](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)
*   [TENETS.md](https://github.com/dbt-labs/metricflow/blob/main/TENETS.md)

## Contributing

We welcome contributions! Please review our [code of conduct](https://docs.getdbt.com/community/resources/code-of-conduct) and refer to the [contributor guide](https://github.com/dbt-labs/metricflow/blob/main/CONTRIBUTING.md) to get started.

## Licensing

MetricFlow is source-available software.

Version 0 to 0.140.0 was covered by the Affero GPL license.
Version 0.150.0 and greater is covered by the BSL license.

MetricFlow is built by [dbt Labs](https://www.getdbt.com/).