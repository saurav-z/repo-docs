<p align="center">
  <a target="_blank" href="https://transform.co/metricflow">
    <picture>
      <img  alt="metricflow logo" src="https://github.com/dbt-labs/metricflow/raw/main/assets/MetricFlow_logo.png" width="auto" height="120">
    </picture>
  </a>
  <br /><br />
</p>

# MetricFlow: Build and Manage Metrics in Code for Consistent Data Insights

**MetricFlow** is a powerful semantic layer that allows you to define and manage your metrics in code, ensuring consistent and reliable data analysis across your organization. Learn more and contribute on [GitHub](https://github.com/dbt-labs/metricflow).

[![Twitter](https://img.shields.io/twitter/follow/dbt_labs?labelColor=image.png&color=163B36&logo=twitter&style=flat)](https://twitter.com/dbt_labs)
[![Slack](https://img.shields.io/badge/Slack-join-163B36)](https://www.getdbt.com/community/)
[![GitHub Stars](https://img.shields.io/github/stars/dbt-labs/metricflow?labelColor=image.png&color=163B36&logo=github)](https://github.com/dbt-labs/metricflow)
[![License](https://img.shields.io/pypi/l/metricflow?color=163B36&logo=AGPL-3.0)](https://github.com/dbt-labs/metricflow/blob/master/LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/metricflow?labelColor=&color=163B36)](https://pypi.org/project/metricflow/)
[![Python Versions](https://img.shields.io/pypi/pyversions/metricflow?labelColor=&color=163B36)](https://pypi.org/project/metricflow/)

**Key Features:**

*   **Metric Definition in Code:** Define metrics using code for version control, reusability, and collaboration.
*   **Automated SQL Generation:**  MetricFlow compiles metric definitions into clean, reusable SQL, reducing manual effort.
*   **Consistent Results:** Ensures consistent and accurate results when analyzing data by relevant attributes (dimensions).
*   **Complex Logic Support:** Handles multi-hop joins, complex metric types (ratio, expression, cumulative), and time-series aggregations.

## What is MetricFlow?

MetricFlow transforms how you manage your metrics.  By defining your metrics in code, you can build a single source of truth for your business metrics, making them reliable and easily accessible for everyone.  MetricFlow's dataflow-based query plan approach optimizes and translates your metric definitions into engine-specific SQL for efficient data retrieval.  Explore the latest updates in the [MetricFlow Changelog](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)!

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

MetricFlow works seamlessly with dbt projects.  Ensure you have a working dbt project and adapter.  Install `dbt-metricflow` and other adapter dependencies as needed. You may also need to install Postgres or Graphviz. See the install instructions for [Postgres](https://www.postgresql.org/download/) or [Graphviz](https://www.graphviz.org/download/). Mac users may prefer to use brew: `brew install postgresql` or `brew install graphviz`.

### Tutorial

Run the tutorial within your dbt project root directory:

```bash
mf tutorial
```

For a complete guide, consult the [MetricFlow documentation](https://docs.getdbt.com/docs/build/build-metrics-intro).

## Licensing

MetricFlow is distributed under a Business Source License (BUSL-1.1). For details on our additional use grant, change license, and change date please refer to our [licensing agreement](https://github.com/dbt-labs/metricflow/blob/main/LICENSE).

## Resources

*   [Website](https://transform.co/metricflow)
*   [Documentation](https://docs.getdbt.com/docs/build/build-metrics-intro)
*   [Slack Community](https://www.getdbt.com/community/)
*   [MetricFlow GitHub Repository](https://github.com/dbt-labs/metricflow)
*   [CHANGELOG.md](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)
*   [TENETS.md](https://github.com/dbt-labs/metricflow/blob/main/TENETS.md)

## Contributing

We welcome contributions! Please review our [code of conduct](https://docs.getdbt.com/community/resources/code-of-conduct) and our [contributor guide](https://github.com/dbt-labs/metricflow/blob/main/CONTRIBUTING.md) to get started.

## About

MetricFlow is built by [dbt Labs](https://www.getdbt.com/).