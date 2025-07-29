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

# MetricFlow: Define, Organize, and Serve Your Metrics as Code

MetricFlow is a powerful semantic layer enabling you to define and manage your business metrics in code, making them consistent, reusable, and easily accessible. **[Visit the MetricFlow GitHub repository for more details.](https://github.com/dbt-labs/metricflow)**

## Key Features

*   **Code-Based Metric Definitions:** Define your metrics directly in code for version control, reusability, and maintainability.
*   **SQL Generation:** Automatically generates clean and efficient SQL code from your metric definitions.
*   **Complex Metric Support:** Handles advanced metric types like ratios, expressions, and cumulative metrics.
*   **Multi-Hop Joins:** Simplifies complex join logic across multiple fact and dimension tables.
*   **Time Granularity Flexibility:** Aggregate metrics at different time granularities.

## How MetricFlow Works

MetricFlow transforms your metric definitions into a query plan, also known as a dataflow, which is then optimized and rendered into engine-specific SQL. This dataflow approach allows for the creation of complex metric logic with ease.

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

MetricFlow integrates with dbt projects. Ensure you have a working dbt project and adapter set up. Refer to the documentation for guidance on installing required dependencies like Postgres and Graphviz.

### Tutorial

Get started with a practical example using the tutorial:

```bash
mf tutorial
```

*This command must be executed from within your dbt project directory.*

## Resources

*   **Website:** [https://transform.co/metricflow](https://transform.co/metricflow)
*   **Documentation:** [https://docs.getdbt.com/docs/build/build-metrics-intro](https://docs.getdbt.com/docs/build/build-metrics-intro)
*   **Slack Community:** [https://www.getdbt.com/community/](https://www.getdbt.com/community/)
*   **MetricFlow GitHub Repository:** [https://github.com/dbt-labs/metricflow](https://github.com/dbt-labs/metricflow)
*   **Changelog:** [https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)
*   **TENETS.md:** [https://github.com/dbt-labs/metricflow/blob/main/TENETS.md](https://github.com/dbt-labs/metricflow/blob/main/TENETS.md)

## Contributing

We welcome contributions! Please review our [code of conduct](https://docs.getdbt.com/community/resources/code-of-conduct) and [contributor guide](https://github.com/dbt-labs/metricflow/blob/main/CONTRIBUTING.md) before getting started.

## License

MetricFlow is source-available software. Version 0 to 0.140.0 was covered by the Affero GPL license. Version 0.150.0 and greater is covered by the BSL license. MetricFlow is built by [dbt Labs](https://www.getdbt.com/).