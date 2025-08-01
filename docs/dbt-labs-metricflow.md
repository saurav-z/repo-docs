<p align="center">
  <a target="_blank" href="https://transform.co/metricflow">
    <picture>
      <img  alt="metricflow logo" src="https://github.com/dbt-labs/metricflow/raw/main/assets/MetricFlow_logo.png" width="auto" height="120">
    </picture>
  </a>
  <br /><br />
</p>

<div align="center">
  <a href="https://twitter.com/dbt_labs" target="_blank">
    <img src="https://img.shields.io/twitter/follow/dbt_labs?labelColor=image.png&color=163B36&logo=twitter&style=flat" alt="Twitter">
  </a>
  <a href="https://www.getdbt.com/community/" target="_blank">
    <img src="https://img.shields.io/badge/Slack-join-163B36" alt="Slack">
  </a>
  <a href="https://github.com/dbt-labs/metricflow" target="_blank">
    <img src="https://img.shields.io/github/stars/dbt-labs/metricflow?labelColor=image.png&color=163B36&logo=github" alt="GitHub Stars">
  </a>
  <br />
  <a href="https://github.com/dbt-labs/metricflow/blob/master/LICENSE" target="_blank">
    <img src="https://img.shields.io/pypi/l/metricflow?color=163B36&logo=AGPL-3.0" alt="License">
  </a>
  <a href="https://pypi.org/project/metricflow/" target="_blank">
    <img src="https://img.shields.io/pypi/v/metricflow?labelColor=&color=163B36" alt="PyPI Version">
  </a>
  <img src="https://img.shields.io/pypi/pyversions/metricflow?labelColor=&color=163B36" alt="Python Versions">
</div>

# MetricFlow: Define, Build, and Manage Metrics in Code

**MetricFlow is a semantic layer that transforms your metric definitions into reusable and legible SQL, empowering data teams to build and maintain consistent and reliable metrics.**

[View the original repository on GitHub](https://github.com/dbt-labs/metricflow)

## Key Features

*   **Metric Definition in Code:** Define metrics using code for version control, reusability, and collaboration.
*   **SQL Generation:** Automatically generates efficient and optimized SQL queries from your metric definitions.
*   **Multi-hop Joins:** Handles complex joins between fact and dimension tables with ease.
*   **Comprehensive Metric Types:** Supports various metric types like ratio, expression, and cumulative calculations.
*   **Time Granularity Control:** Aggregate metrics at different time granularities for flexible analysis.
*   **Data Flow Optimization:**  Compiles queries into dataflows, which are then optimized and rendered into engine-specific SQL.

## Getting Started

### Installation

Install MetricFlow from PyPi:

```bash
pip install dbt-metricflow
```

MetricFlow integrates with your dbt project. Ensure you have a working dbt project and a dbt adapter configured. Refer to the [dbt documentation](https://docs.getdbt.com/docs/get-started/overview) for information on getting started with dbt.

You may also need to install Postgres or Graphviz. You can do so by following the install instructions for [Postgres](https://www.postgresql.org/download/) or [Graphviz](https://www.graphviz.org/download/). Mac users may prefer to use brew: `brew install postgresql` or `brew install graphviz`.

### Tutorial

Get hands-on with MetricFlow by running the tutorial:

```bash
mf tutorial
```

*Note: This command must be run from within your dbt project root directory.*

## Licensing

MetricFlow is distributed under a Business Source License (BUSL-1.1).  Refer to the [licensing agreement](https://github.com/dbt-labs/metricflow/blob/main/LICENSE) for details.

## Resources

*   [Website](https://transform.co/metricflow)
*   [Documentation](https://docs.getdbt.com/docs/build/build-metrics-intro)
*   [Slack Community](https://www.getdbt.com/community/)
*   [MetricFlow GitHub Repository](https://github.com/dbt-labs/metricflow)
*   [CHANGELOG.md](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)
*   [TENETS.md](https://github.com/dbt-labs/metricflow/blob/main/TENETS.md)

## Contributing

We welcome contributions! Please review our [code of conduct](https://docs.getdbt.com/community/resources/code-of-conduct) and [contributor guide](https://github.com/dbt-labs/metricflow/blob/main/CONTRIBUTING.md) to get started.

## License

MetricFlow is source-available software.

Version 0 to 0.140.0 was covered by the Affero GPL license.
Version 0.150.0 and greater is covered by the BSL license.

MetricFlow is built by [dbt Labs](https://www.getdbt.com/).