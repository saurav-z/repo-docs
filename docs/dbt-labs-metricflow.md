<p align="center">
  <a target="_blank" href="https://transform.co/metricflow">
    <picture>
      <img  alt="metricflow logo" src="https://github.com/dbt-labs/metricflow/raw/main/assets/MetricFlow_logo.png" width="auto" height="120">
    </picture>
  </a>
  <br /><br />
</p>

# MetricFlow: Build and Manage Metrics in Code for Data Analysis

**MetricFlow empowers data teams to define, manage, and analyze metrics consistently across their entire organization.**

[![Twitter](https://img.shields.io/twitter/follow/dbt_labs?labelColor=image.png&color=163B36&logo=twitter&style=flat)](https://twitter.com/dbt_labs)
[![Slack](https://img.shields.io/badge/Slack-join-163B36)](https://www.getdbt.com/community/)
[![GitHub Stars](https://img.shields.io/github/stars/dbt-labs/metricflow?labelColor=image.png&color=163B36&logo=github)](https://github.com/dbt-labs/metricflow)
[![License](https://img.shields.io/pypi/l/metricflow?color=163B36&logo=AGPL-3.0)](https://github.com/dbt-labs/metricflow/blob/master/LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/metricflow?labelColor=&color=163B36)](https://pypi.org/project/metricflow/)
[![Python Versions](https://img.shields.io/pypi/pyversions/metricflow?labelColor=&color=163B36)](https://pypi.org/project/metricflow/)

> Stay updated with the latest changes in the [MetricFlow Changelog](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md).

## Key Features of MetricFlow

*   **Code-Based Metric Definitions:** Define metrics in code for version control, reusability, and collaboration.
*   **Consistent Metric Logic:** Compile metric definitions into clear, reusable SQL, ensuring consistent results across all analyses.
*   **Multi-Hop Joins:** Easily handle complex joins between fact and dimension tables.
*   **Advanced Metric Types:** Support for ratio, expression, cumulative, and other complex metric types.
*   **Time Granularity Flexibility:** Aggregate metrics at different time granularities for various reporting needs.
*   **Dataflow-Based Query Planning:** MetricFlow utilizes a dataflow-based approach to optimize and translate metric requests into efficient SQL.

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

MetricFlow is designed to work within a dbt project and requires a working dbt project and a dbt adapter.

You may need to install Postgres or Graphviz. You can do so by following the install instructions for [Postgres](https://www.postgresql.org/download/) or [Graphviz](https://www.graphviz.org/download/). Mac users may prefer to use brew: `brew install postgresql` or `brew install graphviz`.

### Tutorial

Get started quickly with the tutorial:

```bash
mf tutorial
```

*Note: This command must be run from a dbt project root directory.*

## Resources

*   **Website:** [https://transform.co/metricflow](https://transform.co/metricflow)
*   **Documentation:** [https://docs.getdbt.com/docs/build/build-metrics-intro](https://docs.getdbt.com/docs/build/build-metrics-intro)
*   **Slack Community:** [https://www.getdbt.com/community/](https://www.getdbt.com/community/)
*   **GitHub Repository:** [https://github.com/dbt-labs/metricflow](https://github.com/dbt-labs/metricflow)
*   **CHANGELOG.md:** [https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)
*   **TENETS.md:** [https://github.com/dbt-labs/metricflow/blob/main/TENETS.md](https://github.com/dbt-labs/metricflow/blob/main/TENETS.md)

## Contributing

We welcome contributions!  Please review our [Code of Conduct](https://docs.getdbt.com/community/resources/code-of-conduct) and refer to our [contributor guide](https://github.com/dbt-labs/metricflow/blob/main/CONTRIBUTING.md) for more information.

## License

MetricFlow is source-available software.

*   Version 0 to 0.140.0: Affero GPL license.
*   Version 0.150.0 and greater: BSL license.

MetricFlow is built by [dbt Labs](https://www.getdbt.com/).