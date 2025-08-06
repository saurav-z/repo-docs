<p align="center">
  <a target="_blank" href="https://transform.co/metricflow">
    <picture>
      <img  alt="metricflow logo" src="https://github.com/dbt-labs/metricflow/raw/main/assets/MetricFlow_logo.png" width="auto" height="120">
    </picture>
  </a>
  <br /><br />
</p>

# MetricFlow: Build and Manage Metrics in Code

**MetricFlow empowers data teams to define and manage metrics as code, ensuring consistency and accuracy across your data analysis.**

Learn more and contribute on the [MetricFlow GitHub Repository](https://github.com/dbt-labs/metricflow).

[<img src="https://img.shields.io/twitter/follow/dbt_labs?labelColor=image.png&color=163B36&logo=twitter&style=flat">](https://twitter.com/dbt_labs)
[<img src="https://img.shields.io/badge/Slack-join-163B36">](https://www.getdbt.com/community/)
[<img src="https://img.shields.io/github/stars/dbt-labs/metricflow?labelColor=image.png&color=163B36&logo=github">](https://github.com/dbt-labs/metricflow)
[<img src="https://img.shields.io/pypi/l/metricflow?color=163B36&logo=AGPL-3.0">](https://github.com/dbt-labs/metricflow/blob/master/LICENSE)
[<img src="https://img.shields.io/pypi/v/metricflow?labelColor=&color=163B36">](https://pypi.org/project/metricflow/)
[<img src="https://img.shields.io/pypi/pyversions/metricflow?labelColor=&color=163B36">](https://pypi.org/project/metricflow/)

## Key Features

*   **Metric Definition as Code:** Define and manage your metrics in a code-first approach for version control and collaboration.
*   **Consistent Results:** Ensures consistent and accurate results by compiling metric definitions into reusable SQL.
*   **Complex Logic Support:** Handles multi-hop joins, complex metric types (ratio, expression, cumulative), and time-series aggregations.
*   **Dataflow-Based Query Optimization:** MetricFlow uses a dataflow-based query plan for optimization and engine-specific SQL translation.

## What is MetricFlow?

MetricFlow is a semantic layer designed to simplify the process of defining and managing metrics. It allows you to define your metrics in code, which are then compiled into clean, reusable SQL. This approach ensures that your data analysis produces consistent and reliable results across your entire organization.

## Getting Started

### Installation

Install MetricFlow using pip:

```bash
pip install dbt-metricflow
```

MetricFlow integrates with a dbt project. You will also likely need to install the dbt adapter for your data warehouse. For example, to work with Postgres, install `dbt-metricflow` and `dbt-postgres`.

### Tutorial

To get started, run the tutorial from within your dbt project:

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

We welcome contributions! Please review our [Code of Conduct](https://docs.getdbt.com/community/resources/code-of-conduct) and [Contributor Guide](https://github.com/dbt-labs/metricflow/blob/main/CONTRIBUTING.md) to get started.

## License

MetricFlow is source-available software, licensed under a Business Source License (BSL-1.1).