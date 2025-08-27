<p align="center">
  <a target="_blank" href="https://transform.co/metricflow">
    <picture>
      <img  alt="MetricFlow Logo" src="https://github.com/dbt-labs/metricflow/raw/main/assets/MetricFlow_logo.png" width="auto" height="120">
    </picture>
  </a>
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

# MetricFlow: Build Consistent Metrics in Code

**MetricFlow empowers data teams to define and manage metrics in code, ensuring consistent and reliable insights across your organization.**

## Key Features

*   **Semantic Layer:** Define metrics and their relationships in a centralized and reusable way.
*   **Automated SQL Generation:**  Compile metric definitions into optimized SQL queries, eliminating manual query writing and reducing errors.
*   **Multi-Hop Joins:** Effortlessly handle complex joins between fact and dimension tables.
*   **Advanced Metric Types:**  Supports ratio, expression, cumulative, and other complex metric calculations.
*   **Time Granularity Control:**  Aggregate metrics at various time granularities for flexible analysis.
*   **Reusability:** Promote consistency and collaboration across your data team.

<p align="center">
  <img src="https://github.com/dbt-labs/metricflow/raw/main/assets/example_plan.svg" height="400" alt="MetricFlow Dataflow Example"/>
</p>

## Getting Started

### Installation

Install MetricFlow using pip:

```bash
pip install dbt-metricflow
```

MetricFlow integrates seamlessly with dbt projects. Ensure you have a dbt project and adapter configured.  Refer to the [documentation](https://docs.getdbt.com/docs/build/build-metrics-intro) for detailed setup instructions.

### Tutorial

Get started quickly with the tutorial by running:

```bash
mf tutorial
```
(Requires a dbt project root directory)

## Resources

*   [**Documentation**](https://docs.getdbt.com/docs/build/build-metrics-intro)
*   [**Website**](https://transform.co/metricflow)
*   [**Slack Community**](https://www.getdbt.com/community/)
*   [**MetricFlow GitHub Repository**](https://github.com/dbt-labs/metricflow)
*   [CHANGELOG.md](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)
*   [TENETS.md](https://github.com/dbt-labs/metricflow/blob/main/TENETS.md)

## Contributing

We welcome contributions!  Please review our [Code of Conduct](https://docs.getdbt.com/community/resources/code-of-conduct) and [Contributor Guide](https://github.com/dbt-labs/metricflow/blob/main/CONTRIBUTING.md) to get started.

## License

MetricFlow is source-available software.

*   Version 0 to 0.140.0 was covered by the Affero GPL license.
*   Version 0.150.0 and greater is covered by the BSL license.

Built by [dbt Labs](https://www.getdbt.com/).