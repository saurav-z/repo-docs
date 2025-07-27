<p align="center">
  <a target="_blank" href="https://transform.co/metricflow">
    <picture>
      <img  alt="MetricFlow Logo" src="https://github.com/dbt-labs/metricflow/raw/main/assets/MetricFlow_logo.png" width="auto" height="120">
    </picture>
  </a>
</p>

## MetricFlow: Build Consistent and Reusable Metrics in Code

**MetricFlow empowers data teams to define, manage, and generate reliable metrics using code, making data analysis more efficient and consistent.**  Explore the [original MetricFlow repository](https://github.com/dbt-labs/metricflow) for the latest updates and contributions.

<p align="center">
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

### Key Features of MetricFlow:

*   **Metric-as-Code:** Define metrics using code for version control, collaboration, and reusability.
*   **SQL Generation:** Automatically generates optimized and legible SQL queries from your metric definitions.
*   **Complex Logic Support:** Handles multi-hop joins, complex metric types (ratio, expression, cumulative), and time-based aggregations.
*   **Consistent Metrics:** Ensures consistent metric definitions across your organization, reducing errors and promoting trust in your data.
*   **Semantic Layer:** Provides a semantic layer on top of your data warehouse, making it easier for business users to access and understand data.

### Getting Started

1.  **Install MetricFlow:**

    ```bash
    pip install dbt-metricflow
    ```

    MetricFlow is designed to work with dbt.  You will need a working dbt project and adapter. Install Postgres and Graphviz if needed.
2.  **Run the Tutorial:**

    ```bash
    mf tutorial
    ```
    (Run this from your dbt project root.)

### Licensing

MetricFlow is distributed under a Business Source License (BUSL-1.1).  See the [licensing agreement](https://github.com/dbt-labs/metricflow/blob/main/LICENSE) for details.

### Resources

*   [Website](https://transform.co/metricflow)
*   [Documentation](https://docs.getdbt.com/docs/build/build-metrics-intro)
*   [Slack Community](https://www.getdbt.com/community/)
*   [MetricFlow Git Repository](https://github.com/dbt-labs/metricflow)
*   [CHANGELOG.md](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)
*   [TENETS.md](https://github.com/dbt-labs/metricflow/blob/main/TENETS.md)

### Contributing

Contribute to the project! Please read the [code of conduct](https://docs.getdbt.com/community/resources/code-of-conduct) and [contributor guide](https://github.com/dbt-labs/metricflow/blob/main/CONTRIBUTING.md) to get started.

### License

MetricFlow is source-available software.

Version 0 to 0.140.0 was covered by the Affero GPL license.
Version 0.150.0 and greater is covered by the BSL license..

MetricFlow is built by [dbt Labs](https://www.getdbt.com/).