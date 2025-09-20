<p align="center">
  <a target="_blank" href="https://transform.co/metricflow">
    <picture>
      <img  alt="MetricFlow Logo" src="https://github.com/dbt-labs/metricflow/raw/main/assets/MetricFlow_logo.png" width="auto" height="120">
    </picture>
  </a>
</p>

[![Twitter](https://img.shields.io/twitter/follow/dbt_labs?labelColor=image.png&color=163B36&logo=twitter&style=flat)](https://twitter.com/dbt_labs)
[![Slack](https://img.shields.io/badge/Slack-join-163B36)](https://www.getdbt.com/community/)
[![GitHub stars](https://img.shields.io/github/stars/dbt-labs/metricflow?labelColor=image.png&color=163B36&logo=github)](https://github.com/dbt-labs/metricflow)
[![License](https://img.shields.io/pypi/l/metricflow?color=163B36&logo=AGPL-3.0)](https://github.com/dbt-labs/metricflow/blob/master/LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/metricflow?labelColor=&color=163B36)](https://pypi.org/project/metricflow/)
[![Python versions](https://img.shields.io/pypi/pyversions/metricflow?labelColor=&color=163B36)](https://pypi.org/project/metricflow/)

# MetricFlow: Build Consistent and Reusable Metrics in Code

**MetricFlow empowers data teams to define and manage metrics effectively, ensuring consistent and accurate data analysis.**

MetricFlow is a semantic layer that compiles metric definitions into clear, reusable SQL. It transforms metric requests into a dataflow-based query plan, optimized and translated into engine-specific SQL.

## Key Features

*   **Define Metrics in Code:** Manage your metric logic with version control, testing, and collaboration.
*   **Consistent Results:** Ensure consistent and accurate results across all your data analyses.
*   **Multi-Hop Joins:** Handles complex joins between fact and dimension sources.
*   **Complex Metric Types:** Supports ratio, expression, cumulative, and other advanced metric types.
*   **Time Granularity:** Aggregate metrics to different time granularities.
*   **Reusable SQL:** Generate clean, reusable SQL code for your data warehouse.

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

MetricFlow works with dbt projects; therefore, ensure you have a working dbt project and adapter.

### Tutorial

Get started with the tutorial by running:

```bash
mf tutorial
```

*Note: This must be run from a dbt project root directory.*

## Resources

*   [Website](https://transform.co/metricflow)
*   [Documentation](https://docs.getdbt.com/docs/build/build-metrics-intro)
*   [Slack Community](https://www.getdbt.com/community/)
*   [MetricFlow GitHub Repository](https://github.com/dbt-labs/metricflow)
*   [CHANGELOG.md](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)
*   [TENETS.md](https://github.com/dbt-labs/metricflow/blob/main/TENETS.md)

## Contributing

We welcome contributions!  Please review our [Code of Conduct](https://docs.getdbt.com/community/resources/code-of-conduct) and [Contributor Guide](https://github.com/dbt-labs/metricflow/blob/main/CONTRIBUTING.md).

## License

MetricFlow is source-available software.

*   Version 0 to 0.140.0 was covered by the Affero GPL license.
*   Version 0.150.0 and greater is covered by the BSL license.

MetricFlow is built by [dbt Labs](https://www.getdbt.com/).