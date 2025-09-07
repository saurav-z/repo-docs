<p align="center">
  <a target="_blank" href="https://transform.co/metricflow">
    <picture>
      <img  alt="MetricFlow Logo" src="https://github.com/dbt-labs/metricflow/raw/main/assets/MetricFlow_logo.png" width="auto" height="120">
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

# MetricFlow: Your Semantic Layer for Consistent and Reusable Metrics

**MetricFlow empowers data teams to define, manage, and analyze metrics with code, ensuring consistency and accuracy across your data ecosystem.**  Check out the [MetricFlow GitHub Repository](https://github.com/dbt-labs/metricflow) for the latest updates.

## Key Features

*   **Metric Definition in Code:** Define your metrics in a clear, reusable, and version-controlled manner.
*   **Automated SQL Generation:** MetricFlow compiles your metric definitions into optimized, engine-specific SQL.
*   **Multi-Hop Joins:**  Handles complex relationships between fact and dimension tables.
*   **Complex Metric Types:** Supports advanced metrics like ratios, expressions, and cumulative calculations.
*   **Time Granularity Control:** Aggregate metrics at different time granularities for versatile analysis.

## Getting Started

### Installation

Install MetricFlow using pip:

```bash
pip install dbt-metricflow
```

MetricFlow is designed to work with a dbt project. You'll need a working dbt project and a dbt adapter.  The `dbt-metricflow` bundle is provided for this purpose. You may need to install:

*   Postgres
*   Graphviz

See the [Postgres](https://www.postgresql.org/download/) and [Graphviz](https://www.graphviz.org/download/) install instructions. Mac users may prefer to use brew: `brew install postgresql` or `brew install graphviz`.

### Tutorial

Get hands-on with MetricFlow using the tutorial:

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

## Contributing

We welcome contributions! Please review our [code of conduct](https://docs.getdbt.com/community/resources/code-of-conduct). Learn how to contribute in our [contributor guide](https://github.com/dbt-labs/metricflow/blob/main/CONTRIBUTING.md).

## License

MetricFlow is source-available software.

Version 0 to 0.140.0 was covered by the Affero GPL license.
Version 0.150.0 and greater is covered by the BSL license.

MetricFlow is built by [dbt Labs](https://www.getdbt.com/).