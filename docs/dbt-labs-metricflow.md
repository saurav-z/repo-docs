<p align="center">
  <a target="_blank" href="https://transform.co/metricflow">
    <picture>
      <img  alt="MetricFlow logo" src="https://github.com/dbt-labs/metricflow/raw/main/assets/MetricFlow_logo.png" width="auto" height="120">
    </picture>
  </a>
  <br /><br />
</p>

# MetricFlow: Build and Manage Metrics in Code

**MetricFlow empowers data teams to define, manage, and consistently calculate key business metrics directly in code, ensuring reliable and reusable data insights.**  Explore the latest updates in the [MetricFlow Changelog](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)!

MetricFlow acts as a semantic layer, simplifying metric definition and management.  It transforms your metric definitions into clean, reusable SQL, guaranteeing consistent and accurate results across your data analysis, even when analyzing by specific attributes (dimensions). MetricFlow employs a dataflow-based query plan compilation approach, optimized and translated into engine-specific SQL.

## Key Features

*   **Define Metrics as Code:** Define your business metrics within your dbt project, ensuring consistency and reusability.
*   **Multi-Hop Joins:** Handles complex joins between fact and dimension tables with ease.
*   **Advanced Metric Types:** Supports a wide range of metric types, including ratios, expressions, and cumulative metrics.
*   **Time Granularity Control:**  Aggregate metrics at various time granularities, providing flexibility in your analysis.
*   **Consistent Results:** Ensure accuracy and reliability in your data analysis.
*   **SQL Compilation:** Translates metric definitions into optimized SQL for efficient query execution.

## Getting Started

### Installation

Install MetricFlow using pip:

```bash
pip install dbt-metricflow
```

MetricFlow functions as a query compilation and SQL rendering library and requires a dbt project and a dbt adapter. We provide the `dbt-metricflow` bundle for this purpose.

You may need to install Postgres or Graphviz. You can do so by following the install instructions for [Postgres](https://www.postgresql.org/download/) or [Graphviz](https://www.graphviz.org/download/). Mac users may prefer to use brew: `brew install postgresql` or `brew install graphviz`.

### Tutorial

Get started quickly with the tutorial by running:

```bash
mf tutorial
```
*Note: This command must be run from your dbt project root directory.*

## Resources

*   **Website:** [https://transform.co/metricflow](https://transform.co/metricflow)
*   **Documentation:** [https://docs.getdbt.com/docs/build/build-metrics-intro](https://docs.getdbt.com/docs/build/build-metrics-intro)
*   **Slack Community:** [https://www.getdbt.com/community/](https://www.getdbt.com/community/)
*   **GitHub Repository:** [https://github.com/dbt-labs/metricflow](https://github.com/dbt-labs/metricflow)
*   **CHANGELOG.md:** [https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)
*   **TENETS.md:** [https://github.com/dbt-labs/metricflow/blob/main/TENETS.md](https://github.com/dbt-labs/metricflow/blob/main/TENETS.md)

## Contributing & Community

This project welcomes contributions and encourages a collaborative environment.  Please review the [code of conduct](https://docs.getdbt.com/community/resources/code-of-conduct) before contributing.  Find detailed instructions in the [contributor guide](https://github.com/dbt-labs/metricflow/blob/main/CONTRIBUTING.md).

## License

MetricFlow is source-available software.

*   Versions 0 to 0.140.0: Affero GPL license.
*   Versions 0.150.0 and later: BSL license.

MetricFlow is built by [dbt Labs](https://www.getdbt.com/).

---
**[Back to the MetricFlow GitHub Repository](https://github.com/dbt-labs/metricflow)**