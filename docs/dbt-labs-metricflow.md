<p align="center">
  <a target="_blank" href="https://transform.co/metricflow">
    <picture>
      <img  alt="MetricFlow logo" src="https://github.com/dbt-labs/metricflow/raw/main/assets/MetricFlow_logo.png" width="auto" height="120">
    </picture>
  </a>
  <br /><br />
  <b>Build and maintain all of your metric logic in code.</b>
  <br /><br />
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

# MetricFlow: Your Semantic Layer for Consistent Metrics

**MetricFlow empowers data teams to define, manage, and consistently apply metrics across their organization, streamlining data analysis and driving informed decision-making.**

[<img src="https://github.com/dbt-labs/metricflow/raw/main/assets/example_plan.svg" height="300" alt="MetricFlow Data Flow Diagram" />](https://github.com/dbt-labs/metricflow)

## Key Features

*   **Centralized Metric Definitions:** Define all your core business metrics in code for version control, collaboration, and reusability.
*   **Consistent Calculations:** Ensures accurate and consistent metric results across all your data analysis, regardless of the attributes (dimensions) being analyzed.
*   **Complex Metric Support:** Handles advanced metric types, including ratios, expressions, and cumulative calculations.
*   **Time Granularity Aggregation:** Aggregate metrics to different time granularities for flexible analysis.
*   **Dataflow-Based Query Planning:**  Compiles metric requests into an optimized dataflow plan, improving query performance.
*   **Multi-Hop Join Handling:** Simplifies queries involving multiple joins between fact and dimension tables.

## Getting Started

### Installation

Install MetricFlow using pip:

```bash
pip install dbt-metricflow
```

*Note*: Requires a working dbt project and a dbt adapter.

### Tutorial

To get started with MetricFlow, run the tutorial from your dbt project root directory:

```bash
mf tutorial
```

### Prerequisites

You may need to install dependencies such as Postgres or Graphviz:

*   **Postgres:** Follow the install instructions for [Postgres](https://www.postgresql.org/download/) or use `brew install postgresql` on macOS.
*   **Graphviz:** Follow the install instructions for [Graphviz](https://www.graphviz.org/download/) or use `brew install graphviz` on macOS.

## Resources

*   **[Documentation](https://docs.getdbt.com/docs/build/build-metrics-intro)** - Comprehensive guide to using MetricFlow.
*   **[Website](https://transform.co/metricflow)** - Learn more about the project.
*   **[Slack Community](https://www.getdbt.com/community/)** - Join the community and ask questions.
*   **[MetricFlow GitHub Repository](https://github.com/dbt-labs/metricflow)** -  Explore the source code and contribute.
*   **[CHANGELOG.md](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)** - See the latest updates and changes.
*   **[TENETS.md](https://github.com/dbt-labs/metricflow/blob/main/TENETS.md)** - Learn about the core principles of MetricFlow.

## Contributing

We welcome contributions! Please read our [code of conduct](https://docs.getdbt.com/community/resources/code-of-conduct) before contributing.  See the [contributor guide](https://github.com/dbt-labs/metricflow/blob/main/CONTRIBUTING.md) for details.

## License

MetricFlow is source-available software.

*   Version 0 to 0.140.0 was covered by the Affero GPL license.
*   Version 0.150.0 and greater is covered by the BSL license.

MetricFlow is built by [dbt Labs](https://www.getdbt.com/).