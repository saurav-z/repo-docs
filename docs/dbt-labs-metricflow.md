<p align="center">
  <a target="_blank" href="https://transform.co/metricflow">
    <picture>
      <img  alt="MetricFlow Logo" src="https://github.com/dbt-labs/metricflow/raw/main/assets/MetricFlow_logo.png" width="auto" height="120">
    </picture>
  </a>
  <br /><br />
  <b>Build and maintain all of your metric logic in code.</b>
</p>

[![Twitter](https://img.shields.io/twitter/follow/dbt_labs?labelColor=image.png&color=163B36&logo=twitter&style=flat)](https://twitter.com/dbt_labs)
[![Slack](https://img.shields.io/badge/Slack-join-163B36)](https://www.getdbt.com/community/)
[![GitHub Stars](https://img.shields.io/github/stars/dbt-labs/metricflow?labelColor=image.png&color=163B36&logo=github)](https://github.com/dbt-labs/metricflow)
[![License](https://img.shields.io/pypi/l/metricflow?color=163B36&logo=AGPL-3.0)](https://github.com/dbt-labs/metricflow/blob/master/LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/metricflow?labelColor=&color=163B36)](https://pypi.org/project/metricflow/)
[![Python Versions](https://img.shields.io/pypi/pyversions/metricflow?labelColor=&color=163B36)](https://pypi.org/project/metricflow/)

# MetricFlow: Define, Manage, and Analyze Your Metrics in Code

**MetricFlow empowers data teams to build and maintain a reliable semantic layer for consistent and accurate metric analysis.** This open-source project simplifies the process of defining metrics, transforming them into reusable SQL, and ensuring data consistency across your organization.

[See the latest updates in the MetricFlow Changelog](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)!

## Key Features

*   **Metric Definition as Code:** Define metrics using a code-first approach for version control, collaboration, and reusability.
*   **Dataflow-Based Query Compilation:**  MetricFlow compiles metric requests into a dataflow-based query plan, optimizing and translating them into engine-specific SQL.
*   **Multi-Hop Joins:** Seamlessly handle complex joins between fact and dimension tables.
*   **Complex Metric Types:** Support for a variety of metric types, including ratio, expression, and cumulative metrics.
*   **Time Granularity Aggregation:** Aggregate metrics to different time granularities for flexible analysis.
*   **Semantic Layer:**  Provides a consistent and reliable semantic layer that ensures that metrics are defined and used consistently throughout the organization.

<p align="center">
<img src="https://github.com/dbt-labs/metricflow/raw/main/assets/example_plan.svg" height="500"/>
<br /><br />
</p>

## Getting Started

### Installation

Install MetricFlow from PyPi:

```bash
pip install dbt-metricflow
```

MetricFlow works with a dbt project and adapter. Install dbt-metricflow and any necessary adapters. Consider installing Postgres or Graphviz.

### Tutorial

Get started with a tutorial by running:

```bash
mf tutorial
```
(Must be run from a dbt project root directory.)

## Resources

*   [**Website**](https://transform.co/metricflow)
*   [**Documentation**](https://docs.getdbt.com/docs/build/build-metrics-intro)
*   [**Slack Community**](https://www.getdbt.com/community/)
*   [**MetricFlow GitHub Repository**](https://github.com/dbt-labs/metricflow)
*   [CHANGELOG.md](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)
*   [TENETS.md](https://github.com/dbt-labs/metricflow/blob/main/TENETS.md)

## Contributing

We welcome contributions! Please read our [code of conduct](https://docs.getdbt.com/community/resources/code-of-conduct) and [contributor guide](https://github.com/dbt-labs/metricflow/blob/main/CONTRIBUTING.md) to get started.

## License

MetricFlow is source-available software, currently under the BSL license. See the [LICENSE](https://github.com/dbt-labs/metricflow/blob/main/LICENSE) for details.

**Built by [dbt Labs](https://www.getdbt.com/).**