<p align="center">
  <a target="_blank" href="https://transform.co/metricflow">
    <picture>
      <img  alt="MetricFlow Logo" src="https://github.com/dbt-labs/metricflow/raw/main/assets/MetricFlow_logo.png" width="auto" height="120">
    </picture>
  </a>
</p>

<br>

[![Twitter](https://img.shields.io/twitter/follow/dbt_labs?labelColor=image.png&color=163B36&logo=twitter&style=flat)](https://twitter.com/dbt_labs)
[![Slack](https://img.shields.io/badge/Slack-join-163B36)](https://www.getdbt.com/community/)
[![GitHub Stars](https://img.shields.io/github/stars/dbt-labs/metricflow?labelColor=image.png&color=163B36&logo=github)](https://github.com/dbt-labs/metricflow)
[![License](https://img.shields.io/pypi/l/metricflow?color=163B36&logo=AGPL-3.0)](https://github.com/dbt-labs/metricflow/blob/master/LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/metricflow?labelColor=&color=163B36)](https://pypi.org/project/metricflow/)
[![Python Versions](https://img.shields.io/pypi/pyversions/metricflow?labelColor=&color=163B36)](https://pypi.org/project/metricflow/)

# MetricFlow: Build and Manage Metrics in Code

**MetricFlow empowers data teams to define, manage, and analyze business metrics with consistency and accuracy by compiling metric definitions into reusable SQL.**

## Key Features

*   **Semantic Layer:** Define and manage metrics in code for a single source of truth.
*   **Automated SQL Generation:** Compiles metric definitions into optimized SQL queries for various data platforms.
*   **Multi-Hop Joins:** Seamlessly handles complex relationships between fact and dimension tables.
*   **Advanced Metric Types:** Supports ratio, expression, cumulative, and other complex metric calculations.
*   **Time Granularity Flexibility:** Aggregate metrics at different time granularities.
*   **Consistent Results:** Ensures accurate and consistent metric calculations across your organization.

## What is MetricFlow?

MetricFlow is a powerful semantic layer designed to simplify how you define, manage, and analyze business metrics. It allows you to write metric definitions in code, which are then compiled into optimized SQL queries. This approach ensures consistency, accuracy, and reusability across your data analysis workflows. MetricFlow's architecture is based on a dataflow-based query plan that is optimized and translated into engine-specific SQL.

## Getting Started

### Installation

Install MetricFlow using pip:

```bash
pip install dbt-metricflow
```

MetricFlow integrates with your existing dbt project. Ensure you have a working dbt project and a dbt adapter configured.

You may need to install Postgres or Graphviz. You can do so by following the install instructions for [Postgres](https://www.postgresql.org/download/) or [Graphviz](https://www.graphviz.org/download/). Mac users may prefer to use brew: `brew install postgresql` or `brew install graphviz`.

### Tutorial

Get started quickly with the tutorial:

```bash
mf tutorial
```

**Note:** Run this command from within your dbt project directory.

## Resources

*   [Website](https://transform.co/metricflow)
*   [Documentation](https://docs.getdbt.com/docs/build/build-metrics-intro)
*   [Slack Community](https://www.getdbt.com/community/)
*   [MetricFlow GitHub Repository](https://github.com/dbt-labs/metricflow)
*   [CHANGELOG.md](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)
*   [TENETS.md](https://github.com/dbt-labs/metricflow/blob/main/TENETS.md)

## Contributing

We welcome contributions!  Please read our [code of conduct](https://docs.getdbt.com/community/resources/code-of-conduct) and check out our [contributor guide](https://github.com/dbt-labs/metricflow/blob/main/CONTRIBUTING.md) to get started.

## License

MetricFlow is source-available software.

Version 0 to 0.140.0 was covered by the Affero GPL license.
Version 0.150.0 and greater is covered by the BSL license.

MetricFlow is built by [dbt Labs](https://www.getdbt.com/).