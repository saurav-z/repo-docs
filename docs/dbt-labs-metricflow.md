<p align="center">
  <a target="_blank" href="https://transform.co/metricflow">
    <picture>
      <img  alt="MetricFlow Logo" src="https://github.com/dbt-labs/metricflow/raw/main/assets/MetricFlow_logo.png" width="auto" height="120">
    </picture>
  </a>
</p>

<br />

# MetricFlow: Build, Manage, and Query Metrics as Code

**MetricFlow empowers data teams to define and manage their metrics in code, ensuring consistency and accuracy across all their data analysis.**

<br />

[![Twitter](https://img.shields.io/twitter/follow/dbt_labs?labelColor=image.png&color=163B36&logo=twitter&style=flat)](https://twitter.com/dbt_labs)
[![Slack](https://img.shields.io/badge/Slack-join-163B36)](https://www.getdbt.com/community/)
[![GitHub Stars](https://img.shields.io/github/stars/dbt-labs/metricflow?labelColor=image.png&color=163B36&logo=github)](https://github.com/dbt-labs/metricflow)
[![License](https://img.shields.io/pypi/l/metricflow?color=163B36&logo=AGPL-3.0)](https://github.com/dbt-labs/metricflow/blob/master/LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/metricflow?labelColor=&color=163B36)](https://pypi.org/project/metricflow/)
[![Python Versions](https://img.shields.io/pypi/pyversions/metricflow?labelColor=&color=163B36)](https://pypi.org/project/metricflow/)

<br />

Explore the latest updates in the [MetricFlow Changelog](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md).

<br />

## Key Features

*   **Define Metrics in Code:** Easily define and manage your metrics using a code-first approach, promoting version control and collaboration.
*   **Semantic Layer:** Simplifies metric definitions by compiling them into clear, reusable SQL.
*   **Consistent Results:** Ensures consistent and accurate results when analyzing data across different dimensions.
*   **Complex Logic Support:** Construct complex metric types, including ratios, expressions, and cumulative metrics.
*   **Multi-Hop Joins:** Handles complex joins between fact and dimension sources.
*   **Time Granularity Aggregation:** Aggregates metrics to different time granularities.
*   **Dataflow-Based Query Optimization:** Compiles metric requests into a dataflow-based query plan for optimized performance and engine-specific SQL translation.

<br />
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

**Note:** MetricFlow is designed to work with a dbt project. You'll need a working dbt project and adapter. The `dbt-metricflow` bundle is provided for this purpose, and you can install additional adapters as optional extras.

You might also need to install Postgres or Graphviz. Follow the installation instructions for [Postgres](https://www.postgresql.org/download/) or [Graphviz](https://www.graphviz.org/download/). Mac users can install them with brew: `brew install postgresql` or `brew install graphviz`.

### Tutorial

The best way to get started is to follow the tutorial steps, which you can access by running:

```bash
mf tutorial
```

**Important:** Run the above command from your dbt project root directory.

<br />

## Resources

*   **[MetricFlow Website](https://transform.co/metricflow)**
*   **[Documentation](https://docs.getdbt.com/docs/build/build-metrics-intro)**
*   **[Slack Community](https://www.getdbt.com/community/)**
*   **[MetricFlow GitHub Repository](https://github.com/dbt-labs/metricflow)**
*   **[CHANGELOG.md](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)**
*   **[TENETS.md](https://github.com/dbt-labs/metricflow/blob/main/TENETS.md)**

<br />

## Contributing

We welcome contributions! Please read our [code of conduct](https://docs.getdbt.com/community/resources/code-of-conduct) before contributing.

Get started with our [contributor guide](https://github.com/dbt-labs/metricflow/blob/main/CONTRIBUTING.md).

<br />

## License

MetricFlow is source-available software.

*   Version 0 to 0.140.0 was covered by the Affero GPL license.
*   Version 0.150.0 and greater is covered by the BSL license.

MetricFlow is built by [dbt Labs](https://www.getdbt.com/).