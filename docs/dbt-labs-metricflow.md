<p align="center">
  <a target="_blank" href="https://transform.co/metricflow">
    <picture>
      <img  alt="metricflow logo" src="https://github.com/dbt-labs/metricflow/raw/main/assets/MetricFlow_logo.png" width="auto" height="120">
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

# MetricFlow: The Semantic Layer for Consistent Metrics

**MetricFlow simplifies metric definitions, transforming them into reusable and legible SQL for consistent and reliable data analysis.**

[View the original repository on GitHub](https://github.com/dbt-labs/metricflow)

## Key Features

*   **Code-Based Metric Definitions:** Define and maintain all your metric logic directly in code.
*   **Generates Reusable SQL:** Transforms metric definitions into easy-to-understand and reusable SQL queries.
*   **Consistent Metrics:**  Ensures consistent metric output across various dimensions and time granularities.
*   **Supports Complex Logic:** Handles multi-hop joins, complex metric types (ratio, expression, cumulative), and time-based aggregations.
*   **Dataflow Compilation:** Compiles queries into an optimized dataflow plan for efficient execution.

## What is MetricFlow?

MetricFlow is a semantic layer designed to streamline the definition and management of metrics. By defining metrics in code, you can ensure consistency, reusability, and reliability in your data analysis. MetricFlow compiles these definitions into efficient SQL queries, allowing you to easily obtain consistent metrics broken down by the attributes (dimensions) you need.

## Getting Started

### Installation

Install MetricFlow from PyPi using pip:

```bash
pip install dbt-metricflow
```

MetricFlow works with a dbt project and adapter.  Install Postgres or Graphviz if you do not have them.

### Tutorial

Run the following command from your dbt project root directory to access the tutorial:

```bash
mf tutorial
```

## Licensing

MetricFlow is distributed under a Business Source License (BUSL-1.1). For details on our additional use grant, change license, and change date please refer to our [licensing agreement](https://github.com/dbt-labs/metricflow/blob/main/LICENSE).

## Resources

*   [Website](https://transform.co/metricflow)
*   [Documentation](https://docs.getdbt.com/docs/build/build-metrics-intro)
*   [Slack Community](https://www.getdbt.com/community/)
*   [MetricFlow Git Repository](https://github.com/dbt-labs/metricflow)
*   [CHANGELOG.md](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)
*   [TENETS.md](https://github.com/dbt-labs/metricflow/blob/main/TENETS.md)

## Contributing

We welcome contributions!  Please review our [code of conduct](https://docs.getdbt.com/community/resources/code-of-conduct) and refer to our [contributor guide](https://github.com/dbt-labs/metricflow/blob/main/CONTRIBUTING.md) for information on how to get started.

## License

MetricFlow is source-available software.

Version 0 to 0.140.0 was covered by the Affero GPL license.
Version 0.150.0 and greater is covered by the BSL license.

MetricFlow is built by [dbt Labs](https://www.getdbt.com/).