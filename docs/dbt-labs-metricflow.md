<p align="center">
  <a target="_blank" href="https://transform.co/metricflow">
    <picture>
      <img  alt="MetricFlow logo" src="https://github.com/dbt-labs/metricflow/raw/main/assets/MetricFlow_logo.png" width="auto" height="120">
    </picture>
  </a>
  <br /><br />
</p>

# MetricFlow: Build and Manage Metrics in Code

**MetricFlow empowers data teams to define, manage, and consistently measure key business metrics with code.**

[![Twitter](https://img.shields.io/twitter/follow/dbt_labs?labelColor=image.png&color=163B36&logo=twitter&style=flat)](https://twitter.com/dbt_labs)
[![Slack](https://img.shields.io/badge/Slack-join-163B36)](https://www.getdbt.com/community/)
[![GitHub Stars](https://img.shields.io/github/stars/dbt-labs/metricflow?labelColor=image.png&color=163B36&logo=github)](https://github.com/dbt-labs/metricflow)
[![License](https://img.shields.io/pypi/l/metricflow?color=163B36&logo=AGPL-3.0)](https://github.com/dbt-labs/metricflow/blob/master/LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/metricflow?labelColor=&color=163B36)](https://pypi.org/project/metricflow/)
[![Python Versions](https://img.shields.io/pypi/pyversions/metricflow?labelColor=&color=163B36)](https://pypi.org/project/metricflow/)

**[View the MetricFlow GitHub Repository](https://github.com/dbt-labs/metricflow)**

MetricFlow is a powerful semantic layer designed to simplify the creation, maintenance, and consistent application of business metrics. It compiles metric definitions into optimized, reusable SQL, ensuring data accuracy across various dimensions and analyses. Check out the [Metricflow Changelog](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md) for the latest updates!

<p align="center">
<img src="https://github.com/dbt-labs/metricflow/raw/main/assets/example_plan.svg" height="500"/>
<br /><br />
</p>

## Key Features

*   **Define Metrics in Code:** Centralize your metric definitions for easy management, version control, and reusability.
*   **Complex Metric Types:** Supports a wide range of metric types, including ratios, expressions, and cumulative metrics.
*   **Multi-Hop Joins:** Seamlessly handles complex joins between fact and dimension tables.
*   **Time Granularity Aggregation:** Aggregate metrics to different time granularities, enabling flexible reporting.
*   **SQL Compilation & Optimization:** Compiles metric definitions into efficient, engine-specific SQL queries.

## Getting Started

### Installation

Install MetricFlow using pip:

```bash
pip install dbt-metricflow
```

MetricFlow integrates with your existing dbt project.  You may need to install Postgres or Graphviz.  Mac users may prefer to use brew: `brew install postgresql` or `brew install graphviz`.

### Tutorial

Get started quickly with the tutorial:

```bash
mf tutorial
```

*Note:  This command must be run from a dbt project root directory.*

## Resources

*   [Website](https://transform.co/metricflow)
*   [Documentation](https://docs.getdbt.com/docs/build/build-metrics-intro)
*   [Slack Community](https://www.getdbt.com/community/)
*   [MetricFlow Git Repository](https://github.com/dbt-labs/metricflow)
*   [CHANGELOG.md](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)
*   [TENETS.md](https://github.com/dbt-labs/metricflow/blob/main/TENETS.md)

## Contributing

We welcome contributions! Please review our [code of conduct](https://docs.getdbt.com/community/resources/code-of-conduct) and [contributor guide](https://github.com/dbt-labs/metricflow/blob/main/CONTRIBUTING.md) for details.

## License

MetricFlow is source-available software.

*   Version 0 to 0.140.0 was covered by the Affero GPL license.
*   Version 0.150.0 and greater is covered by the BSL license.

MetricFlow is built by [dbt Labs](https://www.getdbt.com/).