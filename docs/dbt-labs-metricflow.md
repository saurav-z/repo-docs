<p align="center">
  <a target="_blank" href="https://transform.co/metricflow">
    <picture>
      <img  alt="MetricFlow Logo" src="https://github.com/dbt-labs/metricflow/raw/main/assets/MetricFlow_logo.png" width="auto" height="120">
    </picture>
  </a>
  <br /><br />
</p>

# MetricFlow: Build and Manage Your Metrics in Code

**MetricFlow empowers data teams to define, manage, and consistently apply metrics across their entire organization.**

[![Twitter](https://img.shields.io/twitter/follow/dbt_labs?labelColor=image.png&color=163B36&logo=twitter&style=flat)](https://twitter.com/dbt_labs)
[![Slack](https://img.shields.io/badge/Slack-join-163B36)](https://www.getdbt.com/community/)
[![GitHub Stars](https://img.shields.io/github/stars/dbt-labs/metricflow?labelColor=image.png&color=163B36&logo=github)](https://github.com/dbt-labs/metricflow)
[![License](https://img.shields.io/pypi/l/metricflow?color=163B36&logo=AGPL-3.0)](https://github.com/dbt-labs/metricflow/blob/master/LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/metricflow?labelColor=&color=163B36)](https://pypi.org/project/metricflow/)
[![Python Versions](https://img.shields.io/pypi/pyversions/metricflow?labelColor=&color=163B36)](https://pypi.org/project/metricflow/)

Read the latest updates in the [MetricFlow Changelog](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)!

## Key Features of MetricFlow:

*   **Metric-Driven Approach:** Define metrics as code for version control, reusability, and consistency.
*   **Semantic Layer:** Simplifies defining and managing metrics, ensuring consistent results.
*   **Dataflow-Based Query Compilation:** Converts metric requests into optimized, engine-specific SQL.
*   **Multi-Hop Joins:** Handles complex joins between fact and dimension sources.
*   **Complex Metric Types:** Supports ratio, expression, and cumulative metrics.
*   **Time Granularity Aggregation:** Aggregate metrics at different time intervals.
*   **dbt Integration:** Works seamlessly with dbt projects for a unified data transformation workflow.

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

MetricFlow works with a dbt project and adapter. Install the `dbt-metricflow` bundle. You may need to install Postgres or Graphviz. Mac users may prefer `brew install postgresql` or `brew install graphviz`.

### Tutorial

Get started quickly with the tutorial:

```bash
mf tutorial
```

(Run from your dbt project root directory.)

## Resources

*   [Website](https://transform.co/metricflow)
*   [Documentation](https://docs.getdbt.com/docs/build/build-metrics-intro)
*   [Slack Community](https://www.getdbt.com/community/)
*   [MetricFlow GitHub Repository](https://github.com/dbt-labs/metricflow)
*   [CHANGELOG.md](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)
*   [TENETS.md](https://github.com/dbt-labs/metricflow/blob/main/TENETS.md)

## Contributing

We welcome contributions! Please read our [code of conduct](https://docs.getdbt.com/community/resources/code-of-conduct) and check out our [contributor guide](https://github.com/dbt-labs/metricflow/blob/main/CONTRIBUTING.md).

## License

MetricFlow is source-available software.
Version 0 to 0.140.0 was covered by the Affero GPL license.
Version 0.150.0 and greater is covered by the BSL license.

MetricFlow is built by [dbt Labs](https://www.getdbt.com/).

---

**[Back to the MetricFlow GitHub Repository](https://github.com/dbt-labs/metricflow)**