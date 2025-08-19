<p align="center">
  <a target="_blank" href="https://transform.co/metricflow">
    <picture>
      <img  alt="MetricFlow Logo" src="https://github.com/dbt-labs/metricflow/raw/main/assets/MetricFlow_logo.png" width="auto" height="120">
    </picture>
  </a>
  <br /><br />
</p>

# MetricFlow: Build Consistent and Reusable Metrics in Code

**MetricFlow empowers data teams to define and manage metrics in code, ensuring consistent and accurate data analysis.**

[![Twitter](https://img.shields.io/twitter/follow/dbt_labs?labelColor=image.png&color=163B36&logo=twitter&style=flat)](https://twitter.com/dbt_labs)
[![Slack](https://img.shields.io/badge/Slack-join-163B36)](https://www.getdbt.com/community/)
[![GitHub stars](https://img.shields.io/github/stars/dbt-labs/metricflow?labelColor=image.png&color=163B36&logo=github)](https://github.com/dbt-labs/metricflow)
[![License](https://img.shields.io/pypi/l/metricflow?color=163B36&logo=AGPL-3.0)](https://github.com/dbt-labs/metricflow/blob/master/LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/metricflow?labelColor=&color=163B36)](https://pypi.org/project/metricflow/)
[![Python versions](https://img.shields.io/pypi/pyversions/metricflow?labelColor=&color=163B36)](https://pypi.org/project/metricflow/)

**[See the latest updates in the MetricFlow Changelog](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)**

## Key Features

*   **Define Metrics in Code:**  Write reusable and maintainable metric definitions.
*   **Semantic Layer:**  Provides a central place to define and manage your key business metrics.
*   **Dataflow-Based Query Plan:** Compiles metric requests into optimized SQL queries.
*   **Multi-Hop Joins:**  Handles complex joins between fact and dimension tables.
*   **Complex Metric Types:** Supports ratio, expression, and cumulative metrics.
*   **Time Granularity Aggregation:**  Aggregate metrics at different time intervals.
*   **Ensure Consistent Results:**  Guarantee accuracy and consistency across all analyses.

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

MetricFlow works in conjunction with a dbt project and adapter.  For more detailed installation and setup instructions, consult the [MetricFlow documentation](https://docs.getdbt.com/docs/build/build-metrics-intro).

### Tutorial

Get started quickly with our interactive tutorial:

```bash
mf tutorial
```

Note: This command must be run from a dbt project root directory.

## Resources

*   [Website](https://transform.co/metricflow)
*   [Documentation](https://docs.getdbt.com/docs/build/build-metrics-intro)
*   [Slack Community](https://www.getdbt.com/community/)
*   [MetricFlow GitHub Repository](https://github.com/dbt-labs/metricflow)
*   [CHANGELOG.md](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)
*   [TENETS.md](https://github.com/dbt-labs/metricflow/blob/main/TENETS.md)

## Contributing

We welcome contributions!  Please review our [code of conduct](https://docs.getdbt.com/community/resources/code-of-conduct) and our [contributor guide](https://github.com/dbt-labs/metricflow/blob/main/CONTRIBUTING.md) to get started.

## Licensing

MetricFlow is source-available software.

*   Version 0 to 0.140.0 was covered by the Affero GPL license.
*   Version 0.150.0 and greater is covered by the BSL license.

MetricFlow is built by [dbt Labs](https://www.getdbt.com/).