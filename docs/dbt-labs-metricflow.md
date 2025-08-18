<p align="center">
  <a target="_blank" href="https://transform.co/metricflow">
    <picture>
      <img  alt="MetricFlow Logo" src="https://github.com/dbt-labs/metricflow/raw/main/assets/MetricFlow_logo.png" width="auto" height="120">
    </picture>
  </a>
  <br /><br />
  <b>Build and maintain all of your metric logic in code.</b>
  <br /><br />
  <a target="_blank" href="https://twitter.com/dbt_labs">
    <img src="https://img.shields.io/twitter/follow/dbt_labs?labelColor=image.png&color=163B36&logo=twitter&style=flat">
  </a>
  <a target="_blank" href="https://www.getdbt.com/community/">
    <img src="https://img.shields.io/badge/Slack-join-163B36">
  </a>
  <a target="_blank" href="https://github.com/dbt-labs/metricflow">
    <img src="https://img.shields.io/github/stars/dbt-labs/metricflow?labelColor=image.png&color=163B36&logo=github">
  </a>
  <br />
  <a target="_blank" href="https://github.com/dbt-labs/metricflow/blob/master/LICENSE">
    <img src="https://img.shields.io/pypi/l/metricflow?color=163B36&logo=AGPL-3.0">
  </a>
  <a target="_blank" href="https://pypi.org/project/metricflow/">
    <img src="https://img.shields.io/pypi/v/metricflow?labelColor=&color=163B36">
  </a>
  <img src="https://img.shields.io/pypi/pyversions/metricflow?labelColor=&color=163B36">
</p>

# MetricFlow: Build Consistent and Reliable Metrics in Code

MetricFlow is a semantic layer that empowers data teams to define, manage, and analyze metrics with code, ensuring consistency and accuracy across your organization.  **[Explore the MetricFlow repository on GitHub](https://github.com/dbt-labs/metricflow) to get started!**

## Key Features

*   **Code-First Metric Definitions:** Define metrics using code for version control, reusability, and maintainability.
*   **Consistent Results:**  Ensure accurate and reliable metrics by compiling definitions into reusable SQL.
*   **Complex Metric Support:**  Handles complex metric types, including ratios, expressions, and cumulative metrics.
*   **Flexible Aggregation:** Aggregate metrics across various time granularities to meet your specific reporting needs.
*   **Multi-Hop Joins:** Simplify complex data relationships with support for multi-hop joins between fact and dimension sources.
*   **Dataflow-Based Query Optimization:** Metric requests are compiled into dataflow-based query plans, optimized, and translated into engine-specific SQL for efficient execution.

## Getting Started

### Installation

Install MetricFlow using pip:

```bash
pip install dbt-metricflow
```

MetricFlow is designed to work with dbt projects. You'll need a working dbt project and a dbt adapter. The `dbt-metricflow` package is provided for this purpose.

### Tutorial

To get started quickly, run the tutorial:

```bash
mf tutorial
```
Note: This command must be run from within a dbt project root directory.

## Resources

*   [Documentation](https://docs.getdbt.com/docs/build/build-metrics-intro)
*   [Website](https://transform.co/metricflow)
*   [Slack Community](https://www.getdbt.com/community/)
*   [CHANGELOG.md](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)
*   [TENETS.md](https://github.com/dbt-labs/metricflow/blob/main/TENETS.md)

## Contributing

We welcome contributions! Please review our [contributor guide](https://github.com/dbt-labs/metricflow/blob/main/CONTRIBUTING.md) and [code of conduct](https://docs.getdbt.com/community/resources/code-of-conduct).

## License

MetricFlow is source-available software, licensed under the Business Source License (BUSL-1.1).

MetricFlow is built by [dbt Labs](https://www.getdbt.com/).