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

# MetricFlow: The Semantic Layer for Consistent and Accurate Metrics

**MetricFlow is a powerful semantic layer that allows you to define, manage, and consistently calculate your most important business metrics in code.** 

<br />

## Key Features

*   **Code-Based Metric Definitions:** Define metrics in code for version control, reusability, and collaboration.
*   **Consistent Calculations:**  Ensures consistent and accurate results across all your data analyses.
*   **Complex Metric Support:** Handles advanced metric types such as ratios, expressions, and cumulative metrics.
*   **Multi-Hop Joins:**  Enables complex joins between fact and dimension tables.
*   **Time Granularity Aggregation:**  Aggregate metrics at different time granularities.
*   **SQL Compilation:** Converts metric definitions into optimized, engine-specific SQL queries.
*   **Dataflow-Based Query Plans:** Utilizes a dataflow approach for efficient query processing.

## Getting Started

### Installation

Install MetricFlow from PyPi using pip:

```bash
pip install dbt-metricflow
```

**Important:** MetricFlow is designed to work with a dbt project and adapter.

You may need to install Postgres or Graphviz. Follow the install instructions for [Postgres](https://www.postgresql.org/download/) or [Graphviz](https://www.graphviz.org/download/). Mac users may prefer to use brew: `brew install postgresql` or `brew install graphviz`.

### Tutorial

Get started quickly with the tutorial:

```bash
mf tutorial
```

**Note:** Run the tutorial from your dbt project's root directory.

## Resources

*   [Website](https://transform.co/metricflow)
*   [Documentation](https://docs.getdbt.com/docs/build/build-metrics-intro)
*   [Slack Community](https://www.getdbt.com/community/)
*   [MetricFlow GitHub Repository](https://github.com/dbt-labs/metricflow)
*   [CHANGELOG.md](https://github.com/dbt-labs/metricflow/blob/main/CHANGELOG.md)
*   [TENETS.md](https://github.com/dbt-labs/metricflow/blob/main/TENETS.md)

## Contributing

We welcome contributions!  Please review our [code of conduct](https://docs.getdbt.com/community/resources/code-of-conduct) and our [contributor guide](https://github.com/dbt-labs/metricflow/blob/main/CONTRIBUTING.md) to get started.

## License

MetricFlow is source-available software, licensed under the Business Source License (BSL-1.1) for versions 0.150.0 and greater.  See the [license](https://github.com/dbt-labs/metricflow/blob/main/LICENSE) for more details.

MetricFlow is built by [dbt Labs](https://www.getdbt.com/).
```
Key improvements and SEO considerations:

*   **Clear Title & Hook:**  The title clearly states what MetricFlow is, and the one-sentence hook summarizes the core value proposition.
*   **Keywords:** Uses relevant keywords like "semantic layer," "metrics," "consistent," "accurate," "SQL," and "data analysis" throughout the text.
*   **Structured Headings:**  Uses clear headings and subheadings for readability and SEO benefits.
*   **Bulleted Key Features:** Highlights the core benefits in an easy-to-scan bulleted list, improving user experience and readability.
*   **Concise Language:** Rephrases the original text to be more direct and action-oriented.
*   **Calls to Action:** Encourages users to get started with the tutorial.
*   **Internal Links:** Includes internal links to important sections like documentation, contributing guidelines, and license.
*   **External Links:**  Provides external links to relevant resources (website, community, GitHub).
*   **Link back to original repo:** The link is still maintained at the resources section for users to be able to access the original repo.
*   **License Information:**  Clearly states the licensing details.
*   **Concise Installation Instructions:** Streamlined installation instructions.
*   **Image Optimization:**  No image optimization was necessary as it was already implemented.