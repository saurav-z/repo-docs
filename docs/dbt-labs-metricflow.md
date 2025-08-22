<p align="center">
  <a target="_blank" href="https://transform.co/metricflow">
    <picture>
      <img  alt="MetricFlow Logo" src="https://github.com/dbt-labs/metricflow/raw/main/assets/MetricFlow_logo.png" width="auto" height="120">
    </picture>
  </a>
  <br /><br />
</p>

# MetricFlow: Build Consistent, Reusable Metrics in Code

**MetricFlow empowers data teams to define and manage metrics in code, ensuring consistent and reliable data analysis.**

[![Twitter](https://img.shields.io/twitter/follow/dbt_labs?labelColor=image.png&color=163B36&logo=twitter&style=flat)](https://twitter.com/dbt_labs)
[![Slack](https://img.shields.io/badge/Slack-join-163B36)](https://www.getdbt.com/community/)
[![GitHub Stars](https://img.shields.io/github/stars/dbt-labs/metricflow?labelColor=image.png&color=163B36&logo=github)](https://github.com/dbt-labs/metricflow)
[![License](https://img.shields.io/pypi/l/metricflow?color=163B36&logo=AGPL-3.0)](https://github.com/dbt-labs/metricflow/blob/master/LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/metricflow?labelColor=&color=163B36)](https://pypi.org/project/metricflow/)
[![Python Version](https://img.shields.io/pypi/pyversions/metricflow?labelColor=&color=163B36)](https://pypi.org/project/metricflow/)

[View the MetricFlow Changelog](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)

MetricFlow is a powerful semantic layer designed to streamline metric definition and management. It translates your metric definitions into clean, reusable SQL, ensuring consistent results across all your data analysis.

MetricFlow uses a dataflow-based query plan, which is optimized and translated into engine-specific SQL.

<p align="center">
<img src="https://github.com/dbt-labs/metricflow/raw/main/assets/example_plan.svg" height="500"/>
<br /><br />
</p>

## Key Features of MetricFlow:

*   **Code-Based Metric Definitions:** Define metrics directly in code for version control and collaboration.
*   **Multi-Hop Joins:** Simplifies complex joins between fact and dimension tables.
*   **Complex Metric Types:** Supports advanced metric types like ratios, expressions, and cumulative metrics.
*   **Time Granularity Aggregation:** Easily aggregates metrics to different time granularities.
*   **Reusable SQL:** Generates clear, optimized SQL queries for consistent results.

## Getting Started

### Installation

Install MetricFlow using pip:

```bash
pip install dbt-metricflow
```

MetricFlow works with a dbt project.

### Tutorial

Get started with MetricFlow by following the tutorial:

```bash
mf tutorial
```
**(Note: Must be run from a dbt project root directory.)**

## Licensing

MetricFlow is distributed under a Business Source License (BUSL-1.1). For details on our additional use grant, change license, and change date please refer to our [licensing agreement](https://github.com/dbt-labs/metricflow/blob/main/LICENSE).

## Resources

*   [Website](https://transform.co/metricflow)
*   [Documentation](https://docs.getdbt.com/docs/build/build-metrics-intro)
*   [Slack Community](https://www.getdbt.com/community/)
*   **[MetricFlow GitHub Repository](https://github.com/dbt-labs/metricflow)**
*   [CHANGELOG.md](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)
*   [TENETS.md](https://github.com/dbt-labs/metricflow/blob/main/TENETS.md)

## Contributing

We welcome contributions!  Please review our [code of conduct](https://docs.getdbt.com/community/resources/code-of-conduct) and refer to the [contributor guide](https://github.com/dbt-labs/metricflow/blob/main/CONTRIBUTING.md) to get started.

## License

MetricFlow is source-available software.

Version 0 to 0.140.0 was covered by the Affero GPL license.
Version 0.150.0 and greater is covered by the BSL license.

MetricFlow is built by [dbt Labs](https://www.getdbt.com/).