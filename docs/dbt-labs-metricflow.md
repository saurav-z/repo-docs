<p align="center">
  <a target="_blank" href="https://transform.co/metricflow">
    <picture>
      <img  alt="MetricFlow Logo" src="https://github.com/dbt-labs/metricflow/raw/main/assets/MetricFlow_logo.png" width="auto" height="120">
    </picture>
  </a>
  <br /><br />
</p>

# MetricFlow: Define, Manage, and Analyze Your Metrics with Code

**MetricFlow empowers data teams to define, manage, and analyze business metrics with a single source of truth.**

MetricFlow is a powerful semantic layer that simplifies the creation and management of metrics, compiling metric definitions into reusable SQL for consistent and accurate data analysis.  This allows for efficient metric creation and management while ensuring consistency across the entire organization.

[Visit the original MetricFlow repository on GitHub](https://github.com/dbt-labs/metricflow)

**Key Features:**

*   **Code-Based Metric Definitions:** Define your metrics in code for version control, reusability, and collaboration.
*   **Simplified Metric Logic:**  Handles complex calculations, including ratios, expressions, and cumulative metrics, with ease.
*   **Multi-Hop Joins:** Efficiently handles complex relationships between fact and dimension tables.
*   **Time Granularity Aggregation:**  Aggregate metrics to various time granularities for in-depth analysis.
*   **Optimized Query Compilation:**  MetricFlow compiles metric requests into a dataflow-based query plan, which is then optimized and translated into engine-specific SQL.
*   **Consistent Results:** Guarantees consistent and accurate metric results across different analyses.

## Getting Started

### Installation

Install MetricFlow using pip:

```bash
pip install dbt-metricflow
```

MetricFlow is designed to work within a dbt project.  Ensure you have a working dbt project and a dbt adapter. The `dbt-metricflow` package provides the necessary components for this.

You may need to install Postgres or Graphviz.  See the original [README](https://github.com/dbt-labs/metricflow) for instructions.

### Tutorial

Start using MetricFlow by running the tutorial:

```bash
mf tutorial
```

*Note: Run this command from your dbt project root directory.*

## Resources

*   [Website](https://transform.co/metricflow)
*   [Documentation](https://docs.getdbt.com/docs/build/build-metrics-intro)
*   [Slack Community](https://www.getdbt.com/community/)
*   [MetricFlow Git Repository](https://github.com/dbt-labs/metricflow)
*   [CHANGELOG.md](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)
*   [TENETS.md](https://github.com/dbt-labs/metricflow/blob/main/TENETS.md)

## Contributing

MetricFlow welcomes contributions from the community.  Please review the [code of conduct](https://docs.getdbt.com/community/resources/code-of-conduct) and the [contributor guide](https://github.com/dbt-labs/metricflow/blob/main/CONTRIBUTING.md).

## License

MetricFlow is source-available software and is distributed under a Business Source License (BUSL-1.1).