<p align="center">
  <a target="_blank" href="https://transform.co/metricflow">
    <picture>
      <img  alt="MetricFlow logo" src="https://github.com/dbt-labs/metricflow/raw/main/assets/MetricFlow_logo.png" width="auto" height="120">
    </picture>
  </a>
  <br /><br />
</p>

# MetricFlow: Build, Manage, and Analyze Metrics with Confidence

**MetricFlow empowers data teams to define and manage their metrics in code, ensuring consistent, accurate, and reusable results.**

[![Twitter](https://img.shields.io/twitter/follow/dbt_labs?labelColor=image.png&color=163B36&logo=twitter&style=flat)](https://twitter.com/dbt_labs)
[![Slack](https://img.shields.io/badge/Slack-join-163B36)](https://www.getdbt.com/community/)
[![GitHub Stars](https://img.shields.io/github/stars/dbt-labs/metricflow?labelColor=image.png&color=163B36&logo=github)](https://github.com/dbt-labs/metricflow)
[![License](https://img.shields.io/pypi/l/metricflow?color=163B36&logo=AGPL-3.0)](https://github.com/dbt-labs/metricflow/blob/master/LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/metricflow?labelColor=&color=163B36)](https://pypi.org/project/metricflow/)
[![Python Versions](https://img.shields.io/pypi/pyversions/metricflow?labelColor=&color=163B36)](https://pypi.org/project/metricflow/)

**Explore the latest updates in the [MetricFlow Changelog](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)!**

MetricFlow is a powerful semantic layer that simplifies the process of defining, managing, and querying your key business metrics. It translates metric definitions into optimized and reusable SQL, guaranteeing consistent and accurate results across all your data analysis.

MetricFlow operates by compiling metric requests into a dataflow-based query plan. This plan is then optimized and transformed into engine-specific SQL for maximum efficiency.

<p align="center">
<img src="https://github.com/dbt-labs/metricflow/raw/main/assets/example_plan.svg" height="500"/>
<br /><br />
</p>

## Key Features of MetricFlow:

*   **Metric Definition in Code:** Define metrics with code for version control, reusability, and collaboration.
*   **Multi-Hop Joins:** Effortlessly handles complex joins between fact and dimension tables.
*   **Advanced Metric Types:** Supports a variety of metric types, including ratios, expressions, and cumulative metrics.
*   **Time Granularity Aggregation:** Aggregates metrics to different time granularities for flexible analysis.
*   **Consistent Results:** Ensures consistent and accurate metric calculations across your entire data ecosystem.
*   **Semantic Layer:** Provides a semantic layer for your data, simplifying data access and improving data literacy.

## Getting Started

### Installation

Install MetricFlow using pip:

```bash
pip install dbt-metricflow
```

MetricFlow integrates seamlessly with dbt projects, acting as a query compilation and SQL rendering library. You'll need a working dbt project and a dbt adapter. The `dbt-metricflow` bundle provides everything you need. Consider installing other adapters as optional extras from dbt-metricflow as well.

You may need to install Postgres or Graphviz. You can do so by following the install instructions for [Postgres](https://www.postgresql.org/download/) or [Graphviz](https://www.graphviz.org/download/). Mac users may prefer to use brew: `brew install postgresql` or `brew install graphviz`.

### Tutorial

To quickly learn the basics, run the tutorial:

```bash
mf tutorial
```

*Note: This command must be executed from your dbt project root directory.*

## Resources

*   [Website](https://transform.co/metricflow)
*   [Documentation](https://docs.getdbt.com/docs/build/build-metrics-intro)
*   [Slack Community](https://www.getdbt.com/community/)
*   [MetricFlow GitHub Repository](https://github.com/dbt-labs/metricflow)
*   [CHANGELOG.md](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)
*   [TENETS.md](https://github.com/dbt-labs/metricflow/blob/main/TENETS.md)

## Contributing

We welcome contributions! Please review our [code of conduct](https://docs.getdbt.com/community/resources/code-of-conduct).

Get started by reading our [contributor guide](https://github.com/dbt-labs/metricflow/blob/main/CONTRIBUTING.md).

## License

MetricFlow is source-available software, licensed under the Business Source License (BUSL-1.1).  See the [LICENSE](https://github.com/dbt-labs/metricflow/blob/master/LICENSE) for details.

*Versions 0 to 0.140.0: Affero GPL License.*
*Versions 0.150.0 and greater: BSL License.*

MetricFlow is developed by [dbt Labs](https://www.getdbt.com/).