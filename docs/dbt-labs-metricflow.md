<p align="center">
  <a target="_blank" href="https://transform.co/metricflow">
    <picture>
      <img  alt="MetricFlow Logo" src="https://github.com/dbt-labs/metricflow/raw/main/assets/MetricFlow_logo.png" width="auto" height="120">
    </picture>
  </a>
</p>

## MetricFlow: Build Consistent, Reusable Metrics in Code

**MetricFlow empowers data teams to define and manage metrics as code, ensuring accuracy and consistency across all your analyses.**

[![Twitter](https://img.shields.io/twitter/follow/dbt_labs?labelColor=image.png&color=163B36&logo=twitter&style=flat)](https://twitter.com/dbt_labs)
[![Slack](https://img.shields.io/badge/Slack-join-163B36)](https://www.getdbt.com/community/)
[![GitHub Stars](https://img.shields.io/github/stars/dbt-labs/metricflow?labelColor=image.png&color=163B36&logo=github)](https://github.com/dbt-labs/metricflow)
[![License](https://img.shields.io/pypi/l/metricflow?color=163B36&logo=AGPL-3.0)](https://github.com/dbt-labs/metricflow/blob/master/LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/metricflow?labelColor=&color=163B36)](https://pypi.org/project/metricflow/)
[![Python Versions](https://img.shields.io/pypi/pyversions/metricflow?labelColor=&color=163B36)](https://pypi.org/project/metricflow/)

**[See the latest updates in the MetricFlow Changelog](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)**

MetricFlow is a semantic layer that transforms how you define and manage your business metrics. It simplifies the process by compiling metric definitions into clear, reusable SQL, guaranteeing consistent and accurate results when analyzing data. MetricFlow's architecture is centered around a dataflow-based query plan, optimized and translated into engine-specific SQL.

<p align="center">
<img src="https://github.com/dbt-labs/metricflow/raw/main/assets/example_plan.svg" height="500"/>
<br /><br />
</p>

### Key Features

*   **Metric Definition as Code:** Define your metrics in a structured, code-based format for version control, reusability, and collaboration.
*   **Multi-Hop Joins:** Handles complex joins between fact and dimension tables, simplifying your data modeling.
*   **Advanced Metric Types:** Supports complex metrics like ratios, expressions, and cumulative calculations.
*   **Time Granularity Control:** Aggregates metrics to different time granularities for flexible analysis.
*   **Dataflow-Based Query Optimization:** Creates an optimized query plan for efficient execution.
*   **SQL Generation:** Translates metric definitions into engine-specific SQL for various data warehouses.

### Getting Started

1.  **Install MetricFlow:**

    ```bash
    pip install dbt-metricflow
    ```

    MetricFlow is designed to work with a dbt project. You'll need a working dbt project and adapter. The `dbt-metricflow` bundle is provided for this purpose. Install optional adapters as needed.
2.  **Prerequisites:** You may need to install Postgres or Graphviz. Instructions are available at:  [Postgres](https://www.postgresql.org/download/) and [Graphviz](https://www.graphviz.org/download/).

3.  **Run the Tutorial:** Get started quickly with the tutorial by running the command from your dbt project root:

    ```bash
    mf tutorial
    ```

### Resources

*   **Website:** [https://transform.co/metricflow](https://transform.co/metricflow)
*   **Documentation:** [https://docs.getdbt.com/docs/build/build-metrics-intro](https://docs.getdbt.com/docs/build/build-metrics-intro)
*   **Slack Community:** [https://www.getdbt.com/community/](https://www.getdbt.com/community/)
*   **GitHub Repository:** [https://github.com/dbt-labs/metricflow](https://github.com/dbt-labs/metricflow)
*   **Changelog:** [https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)
*   **TENETS.md:** [https://github.com/dbt-labs/metricflow/blob/main/TENETS.md](https://github.com/dbt-labs/metricflow/blob/main/TENETS.md)

### Contributing

We welcome contributions! Please review the [Code of Conduct](https://docs.getdbt.com/community/resources/code-of-conduct) and the [Contributor Guide](https://github.com/dbt-labs/metricflow/blob/main/CONTRIBUTING.md) to get started.

### Licensing

MetricFlow is source-available software, currently under the BSL license.
This project is built by [dbt Labs](https://www.getdbt.com/).