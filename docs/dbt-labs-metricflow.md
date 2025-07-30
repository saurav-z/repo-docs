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

# MetricFlow: The Semantic Layer for Consistent and Reusable Metrics

**MetricFlow helps you define, manage, and generate consistent metrics in your data warehouse, simplifying data analysis and empowering data-driven decision-making.**

MetricFlow is a powerful semantic layer designed to streamline your metric definitions and generate reusable SQL. This project, brought to you by dbt Labs, allows you to build and maintain all your metric logic in code.

## Key Features

*   **Code-Based Metric Definitions:** Define your metrics in code for version control, reusability, and collaboration.
*   **Automated SQL Generation:** MetricFlow translates your metric definitions into optimized, engine-specific SQL.
*   **Multi-Hop Join Support:** Handles complex joins between fact and dimension tables with ease.
*   **Advanced Metric Types:** Supports ratio, expression, and cumulative metrics.
*   **Time Granularity Aggregation:**  Aggregate metrics to different time granularities for flexible analysis.

## How It Works

MetricFlow compiles your metric definitions into a query plan, visualized as a dataflow, which is then optimized and rendered into efficient SQL.

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

MetricFlow is designed to work with dbt projects, requiring a working dbt project and adapter. The `dbt-metricflow` bundle is provided for this purpose.  You may need to install additional dependencies such as Postgres or Graphviz, as outlined in the installation instructions.

### Tutorial

Get started quickly with the tutorial:

```bash
mf tutorial
```

*Note: This command must be run from a dbt project root directory.*

## Resources

*   **[Documentation](https://docs.getdbt.com/docs/build/build-metrics-intro)**
*   **[Website](https://transform.co/metricflow)**
*   **[Slack Community](https://www.getdbt.com/community/)**
*   **[MetricFlow GitHub Repository](https://github.com/dbt-labs/metricflow)**
*   [CHANGELOG.md](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)
*   [TENETS.md](https://github.com/dbt-labs/metricflow/blob/main/TENETS.md)

## Contributing and Code of Conduct

We welcome contributions! Please review our [Code of Conduct](https://docs.getdbt.com/community/resources/code-of-conduct) and refer to the [Contributor Guide](https://github.com/dbt-labs/metricflow/blob/main/CONTRIBUTING.md) to get started.

## License

MetricFlow is source-available software, currently under the BSL license (version 0.150.0 and greater).  Refer to the [LICENSE](https://github.com/dbt-labs/metricflow/blob/main/LICENSE) file for complete details.

---
**[Back to the MetricFlow Repository](https://github.com/dbt-labs/metricflow)**