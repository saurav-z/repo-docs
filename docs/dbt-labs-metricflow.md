<p align="center">
  <a target="_blank" href="https://transform.co/metricflow">
    <picture>
      <img  alt="metricflow logo" src="https://github.com/dbt-labs/metricflow/raw/main/assets/MetricFlow_logo.png" width="auto" height="120">
    </picture>
  </a>
</p>

# MetricFlow: Build and Manage Your Metrics in Code

**MetricFlow empowers data teams to define and manage metrics consistently and accurately, simplifying data analysis and ensuring reliable insights.**

[![Twitter](https://img.shields.io/twitter/follow/dbt_labs?labelColor=image.png&color=163B36&logo=twitter&style=flat)](https://twitter.com/dbt_labs)
[![Slack](https://img.shields.io/badge/Slack-join-163B36)](https://www.getdbt.com/community/)
[![GitHub Stars](https://img.shields.io/github/stars/dbt-labs/metricflow?labelColor=image.png&color=163B36&logo=github)](https://github.com/dbt-labs/metricflow)
[![License](https://img.shields.io/pypi/l/metricflow?color=163B36&logo=AGPL-3.0)](https://github.com/dbt-labs/metricflow/blob/master/LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/metricflow?labelColor=&color=163B36)](https://pypi.org/project/metricflow/)
[![Python Versions](https://img.shields.io/pypi/pyversions/metricflow?labelColor=&color=163B36)](https://pypi.org/project/metricflow/)

**Latest Updates:** Check out the [MetricFlow Changelog](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md) to stay up-to-date!

## Key Features

*   **Code-Based Metric Definitions:** Define all your metric logic directly in code for version control, reusability, and maintainability.
*   **Consistent Results:** Ensures consistent and accurate results when analyzing data across different dimensions.
*   **Complex Logic Support:** Handles multi-hop joins, complex metric types (ratios, expressions, cumulative), and time-series aggregations.
*   **Dataflow-Based Query Optimization:** Compiles metric definitions into efficient, optimized SQL queries.
*   **Reusable SQL:** Generates clear, reusable SQL code for consistent analysis.

## Getting Started

### Installation

Install MetricFlow using pip:

```bash
pip install dbt-metricflow
```

MetricFlow integrates seamlessly with dbt projects. You'll need a working dbt project and adapter.  The `dbt-metricflow` bundle provides everything you need.

*Optional Dependencies*: You may need to install Postgres or Graphviz.  For example, on macOS, use `brew install postgresql` and `brew install graphviz`.

### Tutorial

Get started quickly by following the tutorial:

```bash
mf tutorial
```

*   Note: This command must be run from your dbt project's root directory.

## Resources

*   **[MetricFlow Documentation](https://docs.getdbt.com/docs/build/build-metrics-intro)**
*   **[Website](https://transform.co/metricflow)**
*   **[Slack Community](https://www.getdbt.com/community/)**
*   **[MetricFlow GitHub Repository](https://github.com/dbt-labs/metricflow)**
*   [CHANGELOG.md](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)
*   [TENETS.md](https://github.com/dbt-labs/metricflow/blob/main/TENETS.md)

## Contributing

We welcome contributions!  Please review our [code of conduct](https://docs.getdbt.com/community/resources/code-of-conduct) and consult the [contributor guide](https://github.com/dbt-labs/metricflow/blob/main/CONTRIBUTING.md).

## License

MetricFlow is source-available software, licensed under the Business Source License (BUSL-1.1).  See the [license file](https://github.com/dbt-labs/metricflow/blob/main/LICENSE) for details.

MetricFlow is built by [dbt Labs](https://www.getdbt.com/).