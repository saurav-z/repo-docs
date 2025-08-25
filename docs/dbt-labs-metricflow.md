<p align="center">
  <a target="_blank" href="https://transform.co/metricflow">
    <picture>
      <img  alt="MetricFlow logo" src="https://github.com/dbt-labs/metricflow/raw/main/assets/MetricFlow_logo.png" width="auto" height="120">
    </picture>
  </a>
  <br /><br />
</p>

# MetricFlow: The Semantic Layer for Consistent and Accurate Metrics

**MetricFlow empowers data teams to define, manage, and query metrics with code, ensuring a single source of truth for your data.**

[![Twitter](https://img.shields.io/twitter/follow/dbt_labs?labelColor=image.png&color=163B36&logo=twitter&style=flat)](https://twitter.com/dbt_labs)
[![Slack](https://img.shields.io/badge/Slack-join-163B36)](https://www.getdbt.com/community/)
[![GitHub stars](https://img.shields.io/github/stars/dbt-labs/metricflow?labelColor=image.png&color=163B36&logo=github)](https://github.com/dbt-labs/metricflow)
[![License](https://img.shields.io/pypi/l/metricflow?color=163B36&logo=AGPL-3.0)](https://github.com/dbt-labs/metricflow/blob/master/LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/metricflow?labelColor=&color=163B36)](https://pypi.org/project/metricflow/)
[![Python versions](https://img.shields.io/pypi/pyversions/metricflow?labelColor=&color=163B36)](https://pypi.org/project/metricflow/)

**Key Features of MetricFlow:**

*   **Metric Definition in Code:** Define and manage all of your metric logic in code, ensuring consistency and reusability.
*   **Automated SQL Generation:** Compile metric definitions into clear, reusable SQL, streamlining your data analysis process.
*   **Multi-Hop Joins:** Easily handle complex relationships between fact and dimension sources.
*   **Complex Metric Types:** Supports advanced metric types like ratios, expressions, and cumulative metrics.
*   **Time Granularity Aggregation:** Aggregate metrics to different time granularities for flexible analysis.

<p align="center">
<img src="https://github.com/dbt-labs/metricflow/raw/main/assets/example_plan.svg" height="500"/>
<br /><br />
</p>

**Get Started with MetricFlow:**

1.  **Install MetricFlow:**

    ```bash
    pip install dbt-metricflow
    ```
    *Note: MetricFlow works with dbt and requires a working dbt project.*
    *You may need to install Postgres or Graphviz to use metricflow.*

2.  **Follow the Tutorial:**  Get up and running with MetricFlow quickly by running the tutorial:

    ```bash
    mf tutorial
    ```
    *This command must be run from a dbt project root directory.*

**Resources:**

*   [Website](https://transform.co/metricflow)
*   [Documentation](https://docs.getdbt.com/docs/build/build-metrics-intro)
*   [Slack Community](https://www.getdbt.com/community/)
*   [MetricFlow GitHub Repository](https://github.com/dbt-labs/metricflow)
*   [CHANGELOG.md](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)
*   [TENETS.md](https://github.com/dbt-labs/metricflow/blob/main/TENETS.md)

**Licensing & Contributions:**

*   **Licensing:** MetricFlow is distributed under a Business Source License (BUSL-1.1). See the [licensing agreement](https://github.com/dbt-labs/metricflow/blob/main/LICENSE) for details.
*   **Contributing:** We welcome contributions!  Please review our [code of conduct](https://docs.getdbt.com/community/resources/code-of-conduct) and consult our [contributor guide](https://github.com/dbt-labs/metricflow/blob/main/CONTRIBUTING.md).

**Built by [dbt Labs](https://www.getdbt.com/).**

---
*For the original repository, visit: [dbt-labs/metricflow](https://github.com/dbt-labs/metricflow)*