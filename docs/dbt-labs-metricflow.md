<p align="center">
  <a target="_blank" href="https://transform.co/metricflow">
    <picture>
      <img  alt="MetricFlow Logo" src="https://github.com/dbt-labs/metricflow/raw/main/assets/MetricFlow_logo.png" width="auto" height="120">
    </picture>
  </a>
</p>

<div align="center">
  <a href="https://twitter.com/dbt_labs" target="_blank">
    <img src="https://img.shields.io/twitter/follow/dbt_labs?labelColor=image.png&color=163B36&logo=twitter&style=flat" alt="Twitter">
  </a>
  <a href="https://www.getdbt.com/community/" target="_blank">
    <img src="https://img.shields.io/badge/Slack-join-163B36" alt="Slack Community">
  </a>
  <a href="https://github.com/dbt-labs/metricflow" target="_blank">
    <img src="https://img.shields.io/github/stars/dbt-labs/metricflow?labelColor=image.png&color=163B36&logo=github" alt="GitHub Stars">
  </a>
  <br/>
  <a href="https://github.com/dbt-labs/metricflow/blob/master/LICENSE" target="_blank">
    <img src="https://img.shields.io/pypi/l/metricflow?color=163B36&logo=AGPL-3.0" alt="License">
  </a>
  <a href="https://pypi.org/project/metricflow/" target="_blank">
    <img src="https://img.shields.io/pypi/v/metricflow?labelColor=&color=163B36" alt="PyPI Version">
  </a>
  <img src="https://img.shields.io/pypi/pyversions/metricflow?labelColor=&color=163B36" alt="Python Versions">
</div>

# MetricFlow: Build, Manage, and Scale Your Metrics with Code

**MetricFlow simplifies data analysis by allowing you to define and manage your metrics in code, ensuring consistency and accuracy across your data projects.**  This semantic layer compiles metric definitions into reusable SQL, which is then optimized and translated into engine-specific SQL, providing a robust and efficient approach to metric management.  [Learn more about MetricFlow on GitHub](https://github.com/dbt-labs/metricflow).

## Key Features:

*   **Code-Based Metric Definitions:** Define metrics in code for version control, reusability, and collaboration.
*   **Multi-Hop Joins:** Effortlessly handle complex relationships between fact and dimension tables.
*   **Advanced Metric Types:**  Support for ratio, expression, cumulative, and other complex metric types.
*   **Time Granularity Aggregation:** Aggregate metrics at different time intervals for flexible analysis.
*   **SQL Generation:**  Automatically generates optimized SQL queries for various data warehouses.
*   **Semantic Layer:** Provides a consistent and reliable source of truth for your business metrics.

## Getting Started

### Installation

Install MetricFlow using pip:

```bash
pip install dbt-metricflow
```

MetricFlow is designed to work seamlessly with dbt projects. You'll need a working dbt project and a dbt adapter.  The `dbt-metricflow` bundle is provided for this purpose, with additional adapters available as optional extras.

You may need to install Postgres or Graphviz. You can do so by following the install instructions for [Postgres](https://www.postgresql.org/download/) or [Graphviz](https://www.graphviz.org/download/). Mac users may prefer to use brew: `brew install postgresql` or `brew install graphviz`.

### Tutorial

Get started quickly with the built-in tutorial:

```bash
mf tutorial
```

Note: This command must be run from your dbt project root directory.

## Resources

*   [Website](https://transform.co/metricflow)
*   [Documentation](https://docs.getdbt.com/docs/build/build-metrics-intro)
*   [Slack Community](https://www.getdbt.com/community/)
*   [MetricFlow GitHub Repository](https://github.com/dbt-labs/metricflow)
*   [CHANGELOG.md](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)
*   [TENETS.md](https://github.com/dbt-labs/metricflow/blob/main/TENETS.md)

## Contributing

We welcome contributions! Please review our [Code of Conduct](https://docs.getdbt.com/community/resources/code-of-conduct) and [Contributor Guide](https://github.com/dbt-labs/metricflow/blob/main/CONTRIBUTING.md) to get started.

## License

MetricFlow is source-available software.

*   Versions 0 to 0.140.0 were licensed under the Affero GPL.
*   Versions 0.150.0 and greater are licensed under the BSL.

MetricFlow is built by [dbt Labs](https://www.getdbt.com/).