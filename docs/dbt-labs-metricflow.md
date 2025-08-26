<p align="center">
  <a target="_blank" href="https://transform.co/metricflow">
    <picture>
      <img  alt="MetricFlow logo" src="https://github.com/dbt-labs/metricflow/raw/main/assets/MetricFlow_logo.png" width="auto" height="120">
    </picture>
  </a>
  <br /><br />
</p>

# MetricFlow: The Semantic Layer for Consistent and Reliable Metrics

**MetricFlow empowers you to define, manage, and track your key business metrics in code, ensuring data consistency and accuracy across your organization.**  [Learn more about MetricFlow on GitHub](https://github.com/dbt-labs/metricflow).

**Key Features of MetricFlow:**

*   **Centralized Metric Definitions:** Define metrics in code for consistency and reusability.
*   **Automated Query Generation:** MetricFlow compiles metric definitions into optimized SQL queries, eliminating manual query writing.
*   **Multi-Hop Joins:** Easily handle complex relationships between fact and dimension tables.
*   **Support for Complex Metric Types:**  Calculate ratios, expressions, cumulative metrics, and more.
*   **Time Granularity Flexibility:** Aggregate metrics at different time intervals for comprehensive analysis.
*   **Data Flow Optimization:**  MetricFlow uses a dataflow-based query plan for efficient execution.

**Getting Started**

1.  **Installation:**

    ```bash
    pip install dbt-metricflow
    ```

    *MetricFlow requires a dbt project and adapter.  Optional installations may be needed for Postgres or Graphviz.*

2.  **Tutorial:**  Get hands-on with MetricFlow by running the tutorial from your dbt project root:

    ```bash
    mf tutorial
    ```

**Resources**

*   [Documentation](https://docs.getdbt.com/docs/build/build-metrics-intro)
*   [Slack Community](https://www.getdbt.com/community/)
*   [MetricFlow GitHub Repository](https://github.com/dbt-labs/metricflow)
*   [Changelog](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)
*   [TENETS.md](https://github.com/dbt-labs/metricflow/blob/main/TENETS.md)

**Licensing**

MetricFlow is distributed under a Business Source License (BUSL-1.1). For details on our additional use grant, change license, and change date please refer to our [licensing agreement](https://github.com/dbt-labs/metricflow/blob/main/LICENSE).

**Contributing**

We welcome contributions!  Please review our [code of conduct](https://docs.getdbt.com/community/resources/code-of-conduct) and [contributor guide](https://github.com/dbt-labs/metricflow/blob/main/CONTRIBUTING.md).

**Built by dbt Labs**