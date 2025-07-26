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

**MetricFlow empowers data teams to define, manage, and consistently generate metrics for data analysis, simplifying complex data transformations and ensuring reliable insights.** ([View the original repo](https://github.com/dbt-labs/metricflow))

## Key Features

*   **Centralized Metric Definitions:** Define all your metric logic in code for maintainability and reusability.
*   **Automated SQL Generation:** Transforms metric definitions into optimized, engine-specific SQL.
*   **Multi-Hop Join Support:** Seamlessly handles complex joins between fact and dimension tables.
*   **Complex Metric Types:** Supports ratio, expression, cumulative, and other advanced metric calculations.
*   **Time Granularity Aggregation:** Aggregate metrics at different time scales for flexible analysis.

## How MetricFlow Works

MetricFlow operates as a semantic layer, taking metric definitions and compiling them into a query plan (dataflow). This plan is then optimized and rendered into engine-specific SQL, ensuring consistent and reliable metric outputs.

<p align="center">
<img src="https://github.com/dbt-labs/metricflow/raw/main/assets/example_plan.svg" height="500"/>
</p>

## Getting Started

### Installation

Install MetricFlow using pip:

```bash
pip install dbt-metricflow
```

MetricFlow works with a dbt project and adapter. Refer to the official [documentation](https://docs.getdbt.com/docs/build/build-metrics-intro) for setup and usage instructions.

### Tutorial

Start by exploring the tutorial:

```bash
mf tutorial
```

This command must be run from a dbt project root directory.

## Resources

*   [Website](https://transform.co/metricflow)
*   [Documentation](https://docs.getdbt.com/docs/build/build-metrics-intro)
*   [Slack Community](https://www.getdbt.com/community/)
*   [MetricFlow Git Repository](https://github.com/dbt-labs/metricflow)
*   [CHANGELOG.md](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)
*   [TENETS.md](https://github.com/dbt-labs/metricflow/blob/main/TENETS.md)

## Contributing

We welcome contributions! Please review our [code of conduct](https://docs.getdbt.com/community/resources/code-of-conduct) and [contributor guide](https://github.com/dbt-labs/metricflow/blob/main/CONTRIBUTING.md).

## License

MetricFlow is source-available software.