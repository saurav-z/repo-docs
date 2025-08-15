<p align="center">
  <a target="_blank" href="https://transform.co/metricflow">
    <picture>
      <img  alt="MetricFlow Logo" src="https://github.com/dbt-labs/metricflow/raw/main/assets/MetricFlow_logo.png" width="auto" height="120">
    </picture>
  </a>
  <br /><br />
</p>

# MetricFlow: Build and Manage Metrics in Code

**MetricFlow empowers data teams to define, manage, and analyze metrics with consistency and accuracy, making it easier than ever to build a reliable data-driven culture.**

[![Twitter](https://img.shields.io/twitter/follow/dbt_labs?labelColor=image.png&color=163B36&logo=twitter&style=flat)](https://twitter.com/dbt_labs)
[![Slack](https://img.shields.io/badge/Slack-join-163B36)](https://www.getdbt.com/community/)
[![GitHub Stars](https://img.shields.io/github/stars/dbt-labs/metricflow?labelColor=image.png&color=163B36&logo=github)](https://github.com/dbt-labs/metricflow)
[![License](https://img.shields.io/pypi/l/metricflow?color=163B36&logo=AGPL-3.0)](https://github.com/dbt-labs/metricflow/blob/master/LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/metricflow?labelColor=&color=163B36)](https://pypi.org/project/metricflow/)
[![Python Versions](https://img.shields.io/pypi/pyversions/metricflow?labelColor=&color=163B36)](https://pypi.org/project/metricflow/)

*See what's new in the [MetricFlow Changelog](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)!*

## Key Features

*   **Metric Definition as Code:** Define metrics directly in code for version control, collaboration, and reusability.
*   **Consistent and Accurate Results:** MetricFlow compiles metric definitions into optimized SQL, ensuring consistent calculations across all analyses.
*   **Dataflow-Based Query Planning:** MetricFlow uses a dataflow approach to build query plans, optimizing and translating them into engine-specific SQL.
*   **Multi-Hop Joins:** Easily handle complex joins between fact and dimension sources.
*   **Support for Complex Metric Types:** Includes ratio, expression, and cumulative metrics.
*   **Time Granularity Aggregation:** Aggregate metrics at different time granularities.

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

**Prerequisites:**

*   A working dbt project
*   A dbt adapter

**Optional Dependencies:** Postgres, Graphviz (install instructions can be found within the original README)

### Tutorial

Follow the tutorial to quickly learn how to use MetricFlow:

```bash
mf tutorial
```

(Run this command from your dbt project root directory.)

## Licensing

MetricFlow is distributed under a Business Source License (BUSL-1.1). For detailed information, please review the [licensing agreement](https://github.com/dbt-labs/metricflow/blob/main/LICENSE).

## Resources

*   **Website:** [https://transform.co/metricflow](https://transform.co/metricflow)
*   **Documentation:** [https://docs.getdbt.com/docs/build/build-metrics-intro](https://docs.getdbt.com/docs/build/build-metrics-intro)
*   **Slack Community:** [https://www.getdbt.com/community/](https://www.getdbt.com/community/)
*   **GitHub Repository:** [https://github.com/dbt-labs/metricflow](https://github.com/dbt-labs/metricflow)
*   **Changelog:** [https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)
*   **TENETS.md:** [https://github.com/dbt-labs/metricflow/blob/main/TENETS.md](https://github.com/dbt-labs/metricflow/blob/main/TENETS.md)

## Contribute

We welcome contributions! Please review our [code of conduct](https://docs.getdbt.com/community/resources/code-of-conduct) and [contributor guide](https://github.com/dbt-labs/metricflow/blob/main/CONTRIBUTING.md).

## License

MetricFlow is source-available software.

Version 0 to 0.140.0 was covered by the Affero GPL license.
Version 0.150.0 and greater is covered by the BSL license.

MetricFlow is built by [dbt Labs](https://www.getdbt.com/).