<p align="center">
  <a target="_blank" href="https://transform.co/metricflow">
    <picture>
      <img  alt="MetricFlow Logo" src="https://github.com/dbt-labs/metricflow/raw/main/assets/MetricFlow_logo.png" width="auto" height="120">
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

MetricFlow empowers data teams to define and manage metrics in code, ensuring accuracy and consistency across all your data analysis.  [See the original repository](https://github.com/dbt-labs/metricflow).

## Key Features

*   **Metric Definition in Code:** Define your metrics in a clean, reusable, and version-controlled manner.
*   **Consistent Results:**  Ensure accurate and consistent results when analyzing data across different dimensions.
*   **Multi-Hop Joins:** Handles complex joins between fact and dimension tables.
*   **Complex Metric Types:** Supports ratio, expression, and cumulative metrics.
*   **Time Granularity Aggregation:** Aggregate metrics to different time granularities.
*   **Dataflow-Based Query Optimization:** Compiles metric requests into a dataflow-based query plan for optimization and efficient SQL generation.

## Getting Started

### Installation

Install MetricFlow using pip:

```bash
pip install dbt-metricflow
```

MetricFlow integrates with your existing dbt project, serving as a query compilation and SQL rendering library. You'll need a working dbt project and adapter. The `dbt-metricflow` bundle is provided for this purpose and you can install other adapters as optional extras.

### Dependencies

You may need to install Postgres or Graphviz. You can do so by following the install instructions for [Postgres](https://www.postgresql.org/download/) or [Graphviz](https://www.graphviz.org/download/). Mac users may prefer to use brew: `brew install postgresql` or `brew install graphviz`.

### Tutorial

The best way to start using MetricFlow is through the tutorial. Run the following command within your dbt project root directory:

```bash
mf tutorial
```

## Resources

*   [Website](https://transform.co/metricflow)
*   [Documentation](https://docs.getdbt.com/docs/build/build-metrics-intro)
*   [Slack Community](https://www.getdbt.com/community/)
*   [MetricFlow Git Repository](https://github.com/dbt-labs/metricflow)
*   [CHANGELOG.md](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)
*   [TENETS.md](https://github.com/dbt-labs/metricflow/blob/main/TENETS.md)

## Contributing and Code of Conduct

We encourage contributions! Please review our [code of conduct](https://docs.getdbt.com/community/resources/code-of-conduct) and [contributor guide](https://github.com/dbt-labs/metricflow/blob/main/CONTRIBUTING.md) to get started.

## License

MetricFlow is source-available software.

Version 0 to 0.140.0 was covered by the Affero GPL license.
Version 0.150.0 and greater is covered by the BSL license.

MetricFlow is built by [dbt Labs](https://www.getdbt.com/).
```
Key improvements and explanations:

*   **SEO Optimization:** Added a concise and keyword-rich title "MetricFlow: The Semantic Layer for Consistent and Reusable Metrics".  This includes relevant keywords like "semantic layer" and emphasizes the benefits of the tool.
*   **One-Sentence Hook:**  The first sentence now immediately explains the value proposition: "MetricFlow empowers data teams to define and manage metrics in code, ensuring accuracy and consistency across all your data analysis."  This grabs the reader's attention.
*   **Clear Headings:** Used clear and descriptive headings for each section (Key Features, Getting Started, etc.) for readability and SEO.
*   **Bulleted Key Features:**  Uses bullet points to highlight the main capabilities of MetricFlow, making the information easy to scan.
*   **Concise Language:**  Streamlined the descriptions to be more direct and easier to understand.
*   **Installation & Dependencies:**  Improved the installation instructions by clarifying the necessary environment.  I also added a section for dependencies and included installation instructions for postgres and graphviz, since those are sometimes required.
*   **Resource Links:** Keeps the original links but organized them more clearly.
*   **Link Back to Original Repo:** Includes a prominent link back to the original GitHub repository.
*   **Code Formatting:** Maintained code formatting (e.g., code blocks, bold text) for improved readability.
*   **License Clarification**: Added that Metricflow is source-available software and clarified the license history.