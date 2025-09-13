<p align="center">
  <a target="_blank" href="https://transform.co/metricflow">
    <picture>
      <img  alt="MetricFlow Logo" src="https://github.com/dbt-labs/metricflow/raw/main/assets/MetricFlow_logo.png" width="auto" height="120">
    </picture>
  </a>
</p>

<br /><br />

# MetricFlow: Build and Manage Your Metrics in Code

**Simplify data analysis and ensure consistency with MetricFlow, a semantic layer that empowers you to define, manage, and reuse metrics across your organization.** ([View the original repository](https://github.com/dbt-labs/metricflow))

<br />

[![Twitter Follow](https://img.shields.io/twitter/follow/dbt_labs?labelColor=image.png&color=163B36&logo=twitter&style=flat)](https://twitter.com/dbt_labs)
[![Slack Community](https://img.shields.io/badge/Slack-join-163B36)](https://www.getdbt.com/community/)
[![GitHub Stars](https://img.shields.io/github/stars/dbt-labs/metricflow?labelColor=image.png&color=163B36&logo=github)](https://github.com/dbt-labs/metricflow)
<br />
[![License](https://img.shields.io/pypi/l/metricflow?color=163B36&logo=AGPL-3.0)](https://github.com/dbt-labs/metricflow/blob/master/LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/metricflow?labelColor=&color=163B36)](https://pypi.org/project/metricflow/)
[![Python Version](https://img.shields.io/pypi/pyversions/metricflow?labelColor=&color=163B36)](https://pypi.org/project/metricflow/)

<br />

**Stay up to date with the latest improvements in the [MetricFlow Changelog](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)!**

<br />

## Key Features of MetricFlow

*   **Define Metrics as Code:** Write clean, reusable metric definitions in code for clarity and maintainability.
*   **Consistent Data Analysis:** Generate accurate and consistent results by analyzing data across different dimensions.
*   **Multi-Hop Joins & Complex Logic:**  Handles intricate joins, complex metric types (ratios, expressions, etc.), and time-series aggregations.
*   **Dataflow-Based Query Optimization:**  Metric requests are compiled into a dataflow-based query plan.  The plan is then optimized and translated into engine-specific SQL for efficient execution.
*   **Seamless Integration with dbt:** Designed to work within your existing dbt project for easy implementation.

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

MetricFlow leverages your existing dbt project and adapter.

You may need to install Postgres or Graphviz.

**Installation Instructions:**

*   [Postgres](https://www.postgresql.org/download/)
*   [Graphviz](https://www.graphviz.org/download/)
*   Mac users may prefer `brew install postgresql` or `brew install graphviz`.

### Tutorial

To begin your MetricFlow journey, launch the tutorial:

```bash
mf tutorial
```

(Ensure you run this command from your dbt project root directory.)

## Resources

*   **Website:** [MetricFlow Website](https://transform.co/metricflow)
*   **Documentation:** [MetricFlow Documentation](https://docs.getdbt.com/docs/build/build-metrics-intro)
*   **Slack Community:** [Join the Community](https://www.getdbt.com/community/)
*   **GitHub Repository:** [MetricFlow GitHub](https://github.com/dbt-labs/metricflow)
*   **Changelog:** [CHANGELOG.md](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)
*   **Tenets:** [TENETS.md](https://github.com/dbt-labs/metricflow/blob/main/TENETS.md)

## Contributing

We welcome contributions! Review our [code of conduct](https://docs.getdbt.com/community/resources/code-of-conduct) and [contributor guide](https://github.com/dbt-labs/metricflow/blob/main/CONTRIBUTING.md).

## Licensing

MetricFlow is source-available software.

*   **Versions 0 to 0.140.0:** Covered by the Affero GPL license.
*   **Versions 0.150.0 and greater:** Covered by the BSL license.

MetricFlow is developed by [dbt Labs](https://www.getdbt.com/).