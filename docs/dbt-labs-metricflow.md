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

# MetricFlow: The Semantic Layer for Consistent Metrics

**MetricFlow simplifies metric definitions, ensuring data consistency and accuracy for all your data analysis needs.**

MetricFlow is a powerful semantic layer that allows you to define and manage your business metrics in code, transforming them into clear, reusable SQL for consistent results across all your data explorations.  It compiles metric definitions into a dataflow-based query plan, optimizes it, and translates it into engine-specific SQL.

## Key Features

*   **Metric Definition in Code:** Define metrics using code for version control, reusability, and maintainability.
*   **Multi-Hop Joins:** Effortlessly handle complex joins between fact and dimension tables.
*   **Complex Metric Types:** Support for advanced metric types like ratio, expression, and cumulative metrics.
*   **Time Granularity Aggregation:** Aggregate metrics at different time granularities for flexible analysis.
*   **Data Consistency:** Ensures accurate and consistent metric calculations across your organization.
*   **SQL Generation:** Automatically generates optimized SQL queries tailored to your data warehouse.

## Getting Started

### Installation

Install MetricFlow using pip:

```bash
pip install dbt-metricflow
```

**Prerequisites:**

*   A working dbt project
*   A dbt adapter (e.g., dbt-postgres, dbt-snowflake)

You may need to install Postgres or Graphviz. You can do so by following the install instructions for [Postgres](https://www.postgresql.org/download/) or [Graphviz](https://www.graphviz.org/download/). Mac users may prefer to use brew: `brew install postgresql` or `brew install graphviz`.

### Tutorial

Get started by running the tutorial steps from your dbt project root directory:

```bash
mf tutorial
```

## Resources

*   **[MetricFlow Documentation](https://docs.getdbt.com/docs/build/build-metrics-intro)**
*   [Website](https://transform.co/metricflow)
*   [Slack Community](https://www.getdbt.com/community/)
*   **[MetricFlow GitHub Repository](https://github.com/dbt-labs/metricflow)**
*   [CHANGELOG.md](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)
*   [TENETS.md](https://github.com/dbt-labs/metricflow/blob/main/TENETS.md)

## Contributing

We welcome contributions! Please review our [code of conduct](https://docs.getdbt.com/community/resources/code-of-conduct) and refer to our [contributor guide](https://github.com/dbt-labs/metricflow/blob/main/CONTRIBUTING.md) to get started.

## License

MetricFlow is source-available software.

Version 0 to 0.140.0 was covered by the Affero GPL license.
Version 0.150.0 and greater is covered by the BSL license.

MetricFlow is built by [dbt Labs](https://www.getdbt.com/).