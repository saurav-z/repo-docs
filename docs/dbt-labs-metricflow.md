<p align="center">
  <a target="_blank" href="https://transform.co/metricflow">
    <picture>
      <img  alt="metricflow logo" src="https://github.com/dbt-labs/metricflow/raw/main/assets/MetricFlow_logo.png" width="auto" height="120">
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

# MetricFlow: The Semantic Layer for Consistent and Reusable Metrics

**MetricFlow is the open-source semantic layer that empowers you to define, manage, and generate consistent metrics from your data.**

[![MetricFlow GitHub Repository](https://img.shields.io/badge/GitHub-MetricFlow-163B36?logo=github)](https://github.com/dbt-labs/metricflow)

## Key Features

*   **Centralized Metric Definitions:** Define and store all your metric logic in code for a single source of truth.
*   **SQL Generation:** Automatically generates optimized and reusable SQL queries based on your metric definitions.
*   **Multi-Hop Joins:**  Handles complex relationships between fact and dimension tables.
*   **Advanced Metric Types:** Supports ratio, expression, and cumulative metrics for comprehensive analysis.
*   **Time Granularity Aggregation:**  Aggregate metrics to different time granularities as needed.

## What is MetricFlow?

MetricFlow simplifies metric creation by providing a powerful and flexible semantic layer. It allows you to define metrics once, and reuse them consistently across different analyses and reporting tools.  This is achieved by compiling your metric definitions into a query plan (dataflow), which is then optimized and rendered into engine-specific SQL.

<p align="center">
<img src="https://github.com/dbt-labs/metricflow/raw/main/assets/example_plan.svg" height="500"/>
<br /><br />
</p>

## Getting Started

### Installation

Install MetricFlow from PyPi using pip:

```bash
pip install dbt-metricflow
```

MetricFlow integrates seamlessly with dbt projects. Ensure you have a working dbt project and adapter set up. For additional database support, install the necessary adapter for your chosen database (e.g., `dbt-postgres`, `dbt-snowflake`).

You may need to install Postgres or Graphviz. You can do so by following the install instructions for [Postgres](https://www.postgresql.org/download/) or [Graphviz](https://www.graphviz.org/download/). Mac users may prefer to use brew: `brew install postgresql` or `brew install graphviz`.

### Tutorial

Start your MetricFlow journey with the tutorial:

```bash
mf tutorial
```

*Note: Run this command from your dbt project root directory.*

## Resources

*   [Website](https://transform.co/metricflow)
*   [Documentation](https://docs.getdbt.com/docs/build/build-metrics-intro)
*   [Slack Community](https://www.getdbt.com/community/)
*   [MetricFlow GitHub Repository](https://github.com/dbt-labs/metricflow)
*   [CHANGELOG.md](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)
*   [TENETS.md](https://github.com/dbt-labs/metricflow/blob/main/TENETS.md)

## Contributing

Contribute to the project by reading our [code of conduct](https://docs.getdbt.com/community/resources/code-of-conduct) and contributor guide:  [contributor guide](https://github.com/dbt-labs/metricflow/blob/main/CONTRIBUTING.md).

## License

MetricFlow is source-available software, currently under the BSL license.

MetricFlow is built by [dbt Labs](https://www.getdbt.com/).
```
Key improvements and SEO considerations:

*   **Clear Headline:**  Added a more descriptive headline that includes a keyword ("semantic layer").
*   **SEO-Optimized Hook:** The one-sentence hook provides a clear value proposition.
*   **Bulleted Key Features:**  Easy-to-scan format highlights core benefits.
*   **Keyword Usage:** Incorporated relevant keywords like "semantic layer," "metrics," "SQL generation," and "dbt" throughout.
*   **Concise Language:** Replaced verbose phrasing with more direct and impactful language.
*   **Action-Oriented:** Encourages users to get started with commands and links.
*   **Clear Sectioning:**  Uses headings for better readability and structure, which helps with SEO.
*   **Internal Linking:** Links to key resources within the repository.
*   **External Linking:** Added link to original repo, and relevant external resources.
*   **Clearer Installation instructions**: Added guidance to install adapters.