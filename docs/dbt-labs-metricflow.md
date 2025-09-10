<p align="center">
  <a target="_blank" href="https://transform.co/metricflow">
    <picture>
      <img  alt="metricflow logo" src="https://github.com/dbt-labs/metricflow/raw/main/assets/MetricFlow_logo.png" width="auto" height="120">
    </picture>
  </a>
  <br /><br />
</p>

# MetricFlow: Build and Manage Metrics in Code for Consistent Data Analysis

**MetricFlow empowers data teams to define, manage, and analyze their metrics in code, ensuring consistency and accuracy across all data analyses.**

[![Twitter](https://img.shields.io/twitter/follow/dbt_labs?labelColor=image.png&color=163B36&logo=twitter&style=flat)](https://twitter.com/dbt_labs)
[![Slack](https://img.shields.io/badge/Slack-join-163B36)](https://www.getdbt.com/community/)
[![GitHub Stars](https://img.shields.io/github/stars/dbt-labs/metricflow?labelColor=image.png&color=163B36&logo=github)](https://github.com/dbt-labs/metricflow)
[![License](https://img.shields.io/pypi/l/metricflow?color=163B36&logo=AGPL-3.0)](https://github.com/dbt-labs/metricflow/blob/master/LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/metricflow?labelColor=&color=163B36)](https://pypi.org/project/metricflow/)
[![Python Versions](https://img.shields.io/pypi/pyversions/metricflow?labelColor=&color=163B36)](https://pypi.org/project/metricflow/)

## Key Features

*   **Centralized Metric Definitions:** Define and maintain all your metric logic in a single, accessible location.
*   **Consistent Data Across the Board:**  Ensures consistent results when analyzing data by relevant attributes (dimensions).
*   **Dynamic Query Generation:**  Constructs complex logic and dynamically generates queries.
*   **Multi-Hop Join Support:** Handles complex joins between fact and dimension tables.
*   **Complex Metric Types:** Supports various metric types, including ratio, expression, and cumulative metrics.
*   **Time Granularity Aggregation:** Aggregates metrics to different time granularities for flexible analysis.

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

MetricFlow integrates with dbt projects. You'll need a working dbt project and a dbt adapter. Install the `dbt-metricflow` bundle. You might also need to install additional dependencies like Postgres or Graphviz, based on your environment. Refer to their respective documentation for install instructions.

### Tutorial

Get started quickly by running the tutorial from your dbt project root directory:

```bash
mf tutorial
```

## Resources

*   **[Website](https://transform.co/metricflow)**
*   **[Documentation](https://docs.getdbt.com/docs/build/build-metrics-intro)**
*   **[Slack Community](https://www.getdbt.com/community/)**
*   **[MetricFlow GitHub Repository](https://github.com/dbt-labs/metricflow)**
*   [CHANGELOG.md](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)
*   [TENETS.md](https://github.com/dbt-labs/metricflow/blob/main/TENETS.md)

## Contributing

We welcome contributions!  Please review our [contributor guide](https://github.com/dbt-labs/metricflow/blob/main/CONTRIBUTING.md) and [code of conduct](https://docs.getdbt.com/community/resources/code-of-conduct).

## License

MetricFlow is source-available software, with the following licensing history:

*   Versions 0 to 0.140.0: Affero GPL license
*   Versions 0.150.0 and greater: BSL license

MetricFlow is built by [dbt Labs](https://www.getdbt.com/).