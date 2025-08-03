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

# MetricFlow: Define, Manage, and Query Your Metrics in Code

**MetricFlow empowers data teams to define and manage metrics in code, ensuring consistency, reusability, and accurate data insights.**

[View the original repository on GitHub](https://github.com/dbt-labs/metricflow)

## Key Features

*   **Code-Based Metric Definitions:** Define your metrics using code for version control, collaboration, and reusability.
*   **Consistent Metric Output:** Generate legible and reusable SQL, ensuring consistent metric results across your organization.
*   **Multi-Hop Join Support:** Handles complex joins between fact and dimension sources for comprehensive analysis.
*   **Support for Complex Metric Types:** Includes ratio, expression, and cumulative metrics for advanced calculations.
*   **Time Granularity Aggregation:** Aggregate metrics to different time granularities for flexible reporting.

## How MetricFlow Works

MetricFlow functions as a semantic layer, taking your metric definitions and generating SQL queries. This process involves compiling a query into a dataflow plan that constructs metrics, which is then optimized and rendered into engine-specific SQL.

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

MetricFlow works with a dbt project and adapter. The `dbt-metricflow` bundle is provided for this purpose.  You may need to install Postgres or Graphviz; see the install instructions for [Postgres](https://www.postgresql.org/download/) or [Graphviz](https://www.graphviz.org/download/). Mac users may prefer to use brew: `brew install postgresql` or `brew install graphviz`.

### Tutorial

Get started by following the tutorial steps:

```bash
mf tutorial
```

*Note: This command must be run from a dbt project root directory.*

## Resources

*   [Website](https://transform.co/metricflow)
*   [Documentation](https://docs.getdbt.com/docs/build/build-metrics-intro)
*   [Slack Community](https://www.getdbt.com/community/)
*   [MetricFlow GitHub Repository](https://github.com/dbt-labs/metricflow)
*   [CHANGELOG.md](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)
*   [TENETS.md](https://github.com/dbt-labs/metricflow/blob/main/TENETS.md)

## Contributing and Code of Conduct

We welcome contributions! Please read our [code of conduct](https://docs.getdbt.com/community/resources/code-of-conduct) before contributing.  See our [contributor guide](https://github.com/dbt-labs/metricflow/blob/main/CONTRIBUTING.md) for details on how to contribute.

## License

MetricFlow is source-available software.

*   Version 0 to 0.140.0 was covered by the Affero GPL license.
*   Version 0.150.0 and greater is covered by the BSL license.

MetricFlow is built by [dbt Labs](https://www.getdbt.com/).