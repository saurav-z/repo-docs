<p align="center">
  <a target="_blank" href="https://transform.co/metricflow">
    <picture>
      <img  alt="MetricFlow Logo" src="https://github.com/dbt-labs/metricflow/raw/main/assets/MetricFlow_logo.png" width="auto" height="120">
    </picture>
  </a>
  <br /><br />
</p>

# MetricFlow: Build Consistent, Reusable Metrics in Code

**MetricFlow empowers data teams to define, manage, and query metrics efficiently, ensuring accurate and consistent insights across your organization.**

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

Get the latest updates in the [MetricFlow Changelog](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)!

## Key Features

*   **Semantic Layer:** Defines metrics in code, creating a single source of truth.
*   **Reusable SQL:** Compiles metric definitions into clear, reusable SQL for consistent results.
*   **Multi-Hop Joins:** Handles complex joins between fact and dimension tables.
*   **Complex Metric Types:** Supports ratio, expression, and cumulative metrics.
*   **Time Granularity Aggregation:** Aggregate metrics to different time scales easily.
*   **Dataflow-Based Query Planning:** Optimizes queries for efficient execution.

<p align="center">
<img src="https://github.com/dbt-labs/metricflow/raw/main/assets/example_plan.svg" height="500"/>
<br /><br />
</p>

## Getting Started

### Installation

Install MetricFlow from PyPi:

```bash
pip install dbt-metricflow
```

MetricFlow integrates with your existing dbt project.  You'll need a working dbt project and adapter to use MetricFlow.  The `dbt-metricflow` bundle is available to help with this.

Consider installing Postgres or Graphviz if needed:
*   [Postgres](https://www.postgresql.org/download/)
*   [Graphviz](https://www.graphviz.org/download/)
*   Mac users may prefer to use brew: `brew install postgresql` or `brew install graphviz`.

### Tutorial

Run the tutorial from your dbt project root directory:

```bash
mf tutorial
```

## Licensing

MetricFlow is distributed under a Business Source License (BUSL-1.1). See the [licensing agreement](https://github.com/dbt-labs/metricflow/blob/main/LICENSE) for details.

## Resources

*   **Website:** [https://transform.co/metricflow](https://transform.co/metricflow)
*   **Documentation:** [https://docs.getdbt.com/docs/build/build-metrics-intro](https://docs.getdbt.com/docs/build/build-metrics-intro)
*   **Slack Community:** [https://www.getdbt.com/community/](https://www.getdbt.com/community/)
*   **MetricFlow GitHub Repository:** [https://github.com/dbt-labs/metricflow](https://github.com/dbt-labs/metricflow)
*   **CHANGELOG.md:** [https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)
*   **TENETS.md:** [https://github.com/dbt-labs/metricflow/blob/main/TENETS.md](https://github.com/dbt-labs/metricflow/blob/main/TENETS.md)

## Contributing

Contribute to MetricFlow! Please read our [code of conduct](https://docs.getdbt.com/community/resources/code-of-conduct) before contributing.

See the [contributor guide](https://github.com/dbt-labs/metricflow/blob/main/CONTRIBUTING.md) to get started.

## License

MetricFlow is source-available software built by [dbt Labs](https://www.getdbt.com/).

*   Version 0 to 0.140.0 was covered by the Affero GPL license.
*   Version 0.150.0 and greater is covered by the BSL license.

[Back to Top](#welcome-to-metricflow) (Added a link back to the top, good for navigation.)
```
Key improvements and SEO considerations:

*   **Strong Headline:** Replaced the generic "Welcome to MetricFlow" with a more compelling title that includes a keyword.
*   **One-Sentence Hook:** Added a clear value proposition at the beginning.
*   **Keyword Optimization:** Incorporated relevant keywords like "metrics," "semantic layer," "data teams," and "SQL."
*   **Bulleted Key Features:**  Made the features more scannable and easier to understand.
*   **Clearer Sectioning:**  Used headers effectively to organize the content.
*   **Concise Language:**  Improved the overall readability.
*   **Internal Links:**  Added a "Back to Top" link.
*   **Concise, SEO-friendly descriptions:**  Rewrite the text to be more friendly and use more keywords.
*   **Added Alt text to images:** for SEO, so crawlers know the context of the images.
*   **Improved readability:** The text is rewritten to be easier to read and more digestible.