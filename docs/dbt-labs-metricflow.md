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

# MetricFlow: Your Semantic Layer for Consistent Metrics

**MetricFlow is a powerful semantic layer that transforms how you define, manage, and analyze your most important business metrics.**

MetricFlow simplifies the process of building and maintaining metrics by compiling metric definitions into clear, reusable SQL, ensuring consistent and accurate results. It enables you to analyze data effectively by defining metrics once and reusing them across different attributes (dimensions).

## Key Features

*   **Code-Based Metric Definition:** Define your metrics in code for version control, collaboration, and maintainability.
*   **Multi-Hop Joins:** Easily handle complex data relationships with support for multi-hop joins between fact and dimension sources.
*   **Complex Metric Types:**  Calculate advanced metrics like ratios, expressions, and cumulative values with ease.
*   **Time Granularity:** Aggregate your metrics to different time granularities for flexible analysis.
*   **Dataflow-Based Query Compilation:** MetricFlow's unique approach compiles metric requests into a dataflow-based query plan, optimized for efficient SQL generation.

## Getting Started

### Installation

Install MetricFlow using pip:

```bash
pip install dbt-metricflow
```

MetricFlow is designed to work seamlessly with a dbt project. For more detailed information, refer to [MetricFlow’s documentation](https://docs.getdbt.com/docs/build/build-metrics-intro).

### Tutorial

Start with the tutorial by running:

```bash
mf tutorial
```
*Note: This command must be run from a dbt project root directory.*

## Resources

*   **Website:** [MetricFlow Website](https://transform.co/metricflow)
*   **Documentation:** [MetricFlow Documentation](https://docs.getdbt.com/docs/build/build-metrics-intro)
*   **Slack Community:** [Join the Slack Community](https://www.getdbt.com/community/)
*   **GitHub Repository:** [MetricFlow on GitHub](https://github.com/dbt-labs/metricflow)
*   **Changelog:** [MetricFlow Changelog](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)
*   **TENETS:** [MetricFlow Tenets](https://github.com/dbt-labs/metricflow/blob/main/TENETS.md)

## Contributing

We welcome contributions! Please review our [code of conduct](https://docs.getdbt.com/community/resources/code-of-conduct) before contributing. For detailed guidance, see the [contributor guide](https://github.com/dbt-labs/metricflow/blob/main/CONTRIBUTING.md).

## License

MetricFlow is source-available software.

*   Version 0 to 0.140.0 was covered by the Affero GPL license.
*   Version 0.150.0 and greater is covered by the BSL license.

MetricFlow is built by [dbt Labs](https://www.getdbt.com/).