<p align="center">
  <a target="_blank" href="https://transform.co/metricflow">
    <picture>
      <img  alt="MetricFlow Logo" src="https://github.com/dbt-labs/metricflow/raw/main/assets/MetricFlow_logo.png" width="auto" height="120">
    </picture>
  </a>
  <br /><br />
</p>

# MetricFlow: Build and Manage Your Metrics in Code

**MetricFlow empowers data teams to define, manage, and consistently apply metrics across their entire data stack, simplifying complex data analysis.**

[![Twitter](https://img.shields.io/twitter/follow/dbt_labs?labelColor=image.png&color=163B36&logo=twitter&style=flat)](https://twitter.com/dbt_labs)
[![Slack](https://img.shields.io/badge/Slack-join-163B36)](https://www.getdbt.com/community/)
[![GitHub Stars](https://img.shields.io/github/stars/dbt-labs/metricflow?labelColor=image.png&color=163B36&logo=github)](https://github.com/dbt-labs/metricflow)
[![License](https://img.shields.io/pypi/l/metricflow?color=163B36&logo=AGPL-3.0)](https://github.com/dbt-labs/metricflow/blob/master/LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/metricflow?labelColor=&color=163B36)](https://pypi.org/project/metricflow/)
[![Python Version](https://img.shields.io/pypi/pyversions/metricflow?labelColor=&color=163B36)](https://pypi.org/project/metricflow/)

[View the MetricFlow Changelog](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md) to see the latest updates!

MetricFlow acts as a semantic layer, allowing you to define metrics in code and then compile them into optimized, reusable SQL queries. This ensures consistent and accurate results when analyzing your data. MetricFlow translates metric requests into a dataflow-based query plan, which is optimized and translated into engine-specific SQL.

<p align="center">
<img src="https://github.com/dbt-labs/metricflow/raw/main/assets/example_plan.svg" height="500"/>
<br /><br />
</p>

## Key Features

*   **Metric Definition in Code:** Define your metrics using a code-first approach.
*   **Multi-hop Joins:** Handles complex joins between fact and dimension tables.
*   **Complex Metric Types:** Supports various metric types, including ratios, expressions, and cumulative metrics.
*   **Time Granularity Aggregation:** Aggregate metrics to different time granularities.
*   **Reusable SQL:** Compiles metrics into clear, reusable SQL queries.
*   **Semantic Layer:** Provides a consistent and reliable layer for defining and applying metrics.

## Getting Started

### Installation

Install MetricFlow using pip:

```bash
pip install dbt-metricflow
```

**Requirements:** A working dbt project and dbt adapter are required to utilize MetricFlow.  Consider installing additional adapters as optional extras from `dbt-metricflow`.  You may also need to install [Postgres](https://www.postgresql.org/download/) or [Graphviz](https://www.graphviz.org/download/).

### Tutorial

To start learning how to use MetricFlow, run the tutorial from your dbt project root directory:

```bash
mf tutorial
```

## Resources

*   [Website](https://transform.co/metricflow)
*   [Documentation](https://docs.getdbt.com/docs/build/build-metrics-intro)
*   [Slack Community](https://www.getdbt.com/community/)
*   [MetricFlow GitHub Repository](https://github.com/dbt-labs/metricflow)
*   [CHANGELOG.md](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)
*   [TENETS.md](https://github.com/dbt-labs/metricflow/blob/main/TENETS.md)

## Contributing

We welcome contributions! Please review our [Code of Conduct](https://docs.getdbt.com/community/resources/code-of-conduct) and check out the [contributor guide](https://github.com/dbt-labs/metricflow/blob/main/CONTRIBUTING.md) to get started.

## License

MetricFlow is source-available software.

*   Version 0 to 0.140.0 was covered by the Affero GPL license.
*   Version 0.150.0 and greater is covered by the BSL license.

MetricFlow is built by [dbt Labs](https://www.getdbt.com/).