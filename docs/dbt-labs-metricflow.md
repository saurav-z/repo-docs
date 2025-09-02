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

# MetricFlow: The Semantic Layer for Data Metrics

**MetricFlow empowers data teams to define, manage, and analyze metrics efficiently and consistently, all within a code-first approach.**

MetricFlow is a powerful open-source semantic layer that transforms how you define and use metrics. By compiling metric definitions into reusable SQL, it ensures accuracy and consistency across all your data analysis.

## Key Features of MetricFlow

*   **Code-First Metric Definition:** Define your metrics in code for version control, reusability, and collaboration.
*   **Multi-Hop Join Support:** Easily handle complex relationships between fact and dimension tables.
*   **Complex Metric Types:** Supports advanced metric types like ratios, expressions, and cumulative metrics.
*   **Time Granularity Aggregation:** Aggregate metrics to different time granularities for flexible analysis.
*   **Consistent Results:** Ensures accurate and reliable data analysis across all use cases.

## Getting Started with MetricFlow

### Installation

Install MetricFlow using pip:

```bash
pip install dbt-metricflow
```

MetricFlow integrates seamlessly with your existing dbt project. It acts as a query compilation and SQL rendering library, leveraging your existing dbt setup.

### Tutorial

Get started quickly with the tutorial:

```bash
mf tutorial
```

*Note: This command must be run from within a dbt project root directory.*

## Resources

*   [Website](https://transform.co/metricflow)
*   [Documentation](https://docs.getdbt.com/docs/build/build-metrics-intro)
*   [Slack Community](https://www.getdbt.com/community/)
*   [MetricFlow GitHub Repository](https://github.com/dbt-labs/metricflow)
*   [CHANGELOG.md](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)
*   [TENETS.md](https://github.com/dbt-labs/metricflow/blob/main/TENETS.md)

## Contributing

We welcome contributions! Please review our [code of conduct](https://docs.getdbt.com/community/resources/code-of-conduct) before contributing.
See our [contributor guide](https://github.com/dbt-labs/metricflow/blob/main/CONTRIBUTING.md) to get started.

## License

MetricFlow is source-available software under a Business Source License (BUSL-1.1). Refer to the [LICENSE](https://github.com/dbt-labs/metricflow/blob/main/LICENSE) file for details.

Built by [dbt Labs](https://www.getdbt.com/).