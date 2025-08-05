<p align="center">
  <a target="_blank" href="https://transform.co/metricflow">
    <picture>
      <img  alt="MetricFlow Logo" src="https://github.com/dbt-labs/metricflow/raw/main/assets/MetricFlow_logo.png" width="auto" height="120">
    </picture>
  </a>
  <br /><br />
</p>

# MetricFlow: Build & Manage Your Metrics in Code

**MetricFlow is a powerful semantic layer that simplifies defining and managing metrics, ensuring consistent and accurate data analysis.**  See the [original repository](https://github.com/dbt-labs/metricflow) for more details.

<br />
<p align="center">
<img src="https://github.com/dbt-labs/metricflow/raw/main/assets/example_plan.svg" height="500"/>
</p>
<br />

## Key Features

*   **Consistent Metric Definitions:** Define your metrics in code for reusability and maintainability.
*   **Dataflow-Based Query Compilation:** MetricFlow compiles metric requests into a dataflow-based query plan, which is optimized and translated into engine-specific SQL.
*   **Complex Metric Support:** Handles complex metric types such as ratio, expression, and cumulative calculations.
*   **Multi-Hop Joins:** Facilitates joins between fact and dimension sources.
*   **Time Granularity Aggregation:** Aggregates metrics to different time granularities.

## Getting Started

### Installation

Install MetricFlow using pip:

```bash
pip install dbt-metricflow
```

MetricFlow works in conjunction with a dbt project. You'll need a working dbt project and a dbt adapter. Install `dbt-metricflow` and any optional adapters you may need.

You may need to install Postgres or Graphviz.  See the [original README](https://github.com/dbt-labs/metricflow) for install instructions.

### Tutorial

Get started quickly with the tutorial:

```bash
mf tutorial
```

*Note: This command must be run from a dbt project root directory.*

## Resources

*   [Website](https://transform.co/metricflow)
*   [Documentation](https://docs.getdbt.com/docs/build/build-metrics-intro)
*   [Slack Community](https://www.getdbt.com/community/)
*   [MetricFlow GitHub Repository](https://github.com/dbt-labs/metricflow)
*   [CHANGELOG.md](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)
*   [TENETS.md](https://github.com/dbt-labs/metricflow/blob/main/TENETS.md)

## Contributing

We welcome contributions! Please refer to our [contributor guide](https://github.com/dbt-labs/metricflow/blob/main/CONTRIBUTING.md) and [code of conduct](https://docs.getdbt.com/community/resources/code-of-conduct).

## License

MetricFlow is source-available software under a Business Source License (BUSL-1.1).