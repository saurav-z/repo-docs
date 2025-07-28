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

**MetricFlow empowers you to define and manage your business metrics in code, generating reusable and understandable SQL for consistent insights.** ([View the source on GitHub](https://github.com/dbt-labs/metricflow))

MetricFlow is a powerful semantic layer that simplifies the process of defining, calculating, and accessing key business metrics. By centralizing metric logic, MetricFlow ensures consistency and accuracy across your data analyses.

## Key Features:

*   **Code-Based Metric Definitions:** Define metrics using code for version control, reusability, and collaboration.
*   **Automated SQL Generation:** Automatically generates optimized SQL queries from your metric definitions, saving time and reducing errors.
*   **Multi-Hop Joins:** Handles complex joins between fact and dimension tables with ease.
*   **Complex Metric Types:** Supports a wide range of metric types, including ratios, expressions, and cumulative metrics.
*   **Time Granularity:** Aggregate metrics to different time granularities for flexible reporting.

## Getting Started

### Installation

Install MetricFlow using pip:

```bash
pip install dbt-metricflow
```

MetricFlow integrates seamlessly with dbt projects, leveraging your existing dbt setup and adapter.

**Prerequisites:** You may need to install PostgreSQL or Graphviz. Mac users can install with `brew install postgresql` and `brew install graphviz`.

### Tutorial

Start your MetricFlow journey with a helpful tutorial:

```bash
mf tutorial
```

*Run this command from your dbt project root directory.*

## Resources

*   [Website](https://transform.co/metricflow)
*   [Documentation](https://docs.getdbt.com/docs/build/build-metrics-intro)
*   [Slack Community](https://www.getdbt.com/community/)
*   [MetricFlow Git Repository](https://github.com/dbt-labs/metricflow)
*   [CHANGELOG.md](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)
*   [TENETS.md](https://github.com/dbt-labs/metricflow/blob/main/TENETS.md)

## Contributing and Code of Conduct

We welcome contributions! Please review our [code of conduct](https://docs.getdbt.com/community/resources/code-of-conduct) and [contributor guide](https://github.com/dbt-labs/metricflow/blob/main/CONTRIBUTING.md) before contributing.

## License

MetricFlow is source-available software.

Version 0 to 0.140.0 was covered by the Affero GPL license.
Version 0.150.0 and greater is covered by the BSL license.

Built by [dbt Labs](https://www.getdbt.com/).