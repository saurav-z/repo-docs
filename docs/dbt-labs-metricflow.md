<p align="center">
  <a target="_blank" href="https://transform.co/metricflow">
    <picture>
      <img  alt="MetricFlow logo" src="https://github.com/dbt-labs/metricflow/raw/main/assets/MetricFlow_logo.png" width="auto" height="120">
    </picture>
  </a>
</p>

[![Twitter](https://img.shields.io/twitter/follow/dbt_labs?labelColor=image.png&color=163B36&logo=twitter&style=flat)](https://twitter.com/dbt_labs)
[![Slack](https://img.shields.io/badge/Slack-join-163B36)](https://www.getdbt.com/community/)
[![GitHub stars](https://img.shields.io/github/stars/dbt-labs/metricflow?labelColor=image.png&color=163B36&logo=github)](https://github.com/dbt-labs/metricflow)
[![License](https://img.shields.io/pypi/l/metricflow?color=163B36&logo=AGPL-3.0)](https://github.com/dbt-labs/metricflow/blob/master/LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/metricflow?labelColor=&color=163B36)](https://pypi.org/project/metricflow/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/metricflow?labelColor=&color=163B36)](https://pypi.org/project/metricflow/)

# MetricFlow: Build Consistent and Reusable Metrics in Code

MetricFlow simplifies defining and managing metrics, enabling consistent and accurate data analysis.

**Key Features:**

*   **Semantic Layer:** Define and manage metrics in code for consistency and reusability.
*   **Multi-Hop Joins:** Handles complex joins between fact and dimension tables.
*   **Complex Metric Types:** Supports ratio, expression, and cumulative metrics.
*   **Time Granularity Aggregation:** Aggregate metrics to different time granularities.
*   **Dataflow-Based Query Compilation:** Compiles metric requests into optimized SQL queries.

<p align="center">
<img src="https://github.com/dbt-labs/metricflow/raw/main/assets/example_plan.svg" height="500"/>
</p>

## Getting Started

### Installation

Install MetricFlow using pip:

```bash
pip install dbt-metricflow
```

MetricFlow works with dbt projects. You'll need a working dbt project and adapter.  Install `dbt-metricflow` for this purpose.  Optional dbt adapters can be installed as extras. You may need to install Postgres or Graphviz.

### Tutorial

Get started quickly with the tutorial.  Run this command from your dbt project root directory:

```bash
mf tutorial
```

## Resources

*   [Website](https://transform.co/metricflow)
*   [Documentation](https://docs.getdbt.com/docs/build/build-metrics-intro)
*   [Slack Community](https://www.getdbt.com/community/)
*   [MetricFlow GitHub Repository](https://github.com/dbt-labs/metricflow)
*   [CHANGELOG.md](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)
*   [TENETS.md](https://github.com/dbt-labs/metricflow/blob/main/TENETS.md)

## Contributing

We welcome contributions! Please review our [code of conduct](https://docs.getdbt.com/community/resources/code-of-conduct) and [contributor guide](https://github.com/dbt-labs/metricflow/blob/main/CONTRIBUTING.md).

## License

MetricFlow is source-available software. Details in the [LICENSE](https://github.com/dbt-labs/metricflow/blob/master/LICENSE).

Built by [dbt Labs](https://www.getdbt.com/).