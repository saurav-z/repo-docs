# Apache Airflow: Orchestrate Your Workflows with Ease

**Apache Airflow** is a powerful, open-source platform for programmatically authoring, scheduling, and monitoring complex workflows, making data pipelines manageable and scalable.  [Explore the official Apache Airflow repository](https://github.com/apache/airflow) for the latest updates and contributions.

## Key Features:

*   **Dynamic Workflows:** Define pipelines as code for maintainability, version control, and collaboration.
*   **Extensible Architecture:** Leverage a rich set of built-in operators and customize Airflow to meet your specific needs.
*   **Flexible Scheduling:** Schedule tasks with dependencies, execution times, and customizable parameters.
*   **Robust Monitoring:**  Visualize pipeline progress, monitor performance, and troubleshoot issues through the user-friendly web interface.
*   **Idempotent Tasks:** Designed to support idempotent tasks to avoid creating duplicated data
*   **Jinja Templating:**  Use Jinja templating for rich customizations
*   **User Interface:** Rich Web UI to view DAGs, assets, graphs, home and many more.

## Getting Started:

Dive into Airflow with these resources:

*   [Installation Guide](https://airflow.apache.org/docs/apache-airflow/stable/installation/)
*   [Getting Started Tutorial](https://airflow.apache.org/docs/apache-airflow/stable/start.html)
*   [Comprehensive Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)

## Installation:

For detailed installation instructions, refer to the [INSTALLING.md](INSTALLING.md) file in this repository.

### Installing from PyPI

Install Airflow with the latest stable version using constraints:

```bash
pip install 'apache-airflow==3.0.2' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.2/constraints-3.10.txt"
```

Install with extras, for example, with postgres and google providers:

```bash
pip install 'apache-airflow[postgres,google]==3.0.2' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.2/constraints-3.10.txt"
```

## Official Source Code

Apache Airflow is an [Apache Software Foundation](https://www.apache.org) (ASF) project, and our official source code releases are available at:

*   [ASF Distribution Directory](https://downloads.apache.org/airflow)

## Version Life Cycle

Stay informed about the support status of each Airflow version:

<!-- This table is automatically updated by pre-commit scripts/ci/pre_commit/supported_versions.py -->
<!-- Beginning of auto-generated table -->

| Version   | Current Patch/Minor   | State     | First Release   | Limited Maintenance   | EOL/Terminated   |
|-----------|-----------------------|-----------|-----------------|-----------------------|------------------|
| 3         | 3.0.2                 | Supported | Apr 22, 2025    | TBD                   | TBD              |
| 2         | 2.11.0                | Supported | Dec 17, 2020    | Oct 22, 2025          | Apr 22, 2026     |
| 1.10      | 1.10.15               | EOL       | Aug 27, 2018    | Dec 17, 2020          | June 17, 2021    |
| 1.9       | 1.9.0                 | EOL       | Jan 03, 2018    | Aug 27, 2018          | Aug 27, 2018     |
| 1.8       | 1.8.2                 | EOL       | Mar 19, 2017    | Jan 03, 2018          | Jan 03, 2018     |
| 1.7       | 1.7.1.2               | EOL       | Mar 28, 2016    | Mar 19, 2017          | Mar 19, 2017     |

<!-- End of auto-generated table -->

## Contributing:

Contribute to Airflow!  Learn how in the [contributors' guide](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst) and [contribution quickstart](https://github.com/apache/airflow/blob/main/contributing-docs/03_contributors_quick_start.rst).

## Who Uses Apache Airflow?

Join a community of over 500 organizations that use Apache Airflow ([INTHEWILD.md](https://github.com/apache/airflow/blob/main/INTHEWILD.md)).

## Community and Support

*   [Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
*   [Chat](https://s.apache.org/airflow-slack)
*   [Community Information](https://airflow.apache.org/community/)

## Sponsors

The CI infrastructure for Apache Airflow has been sponsored by:

<!-- Ordered by most recently "funded" -->

<a href="https://astronomer.io"><img src="https://assets2.astronomer.io/logos/logoForLIGHTbackground.png" alt="astronomer.io" width="250px"></a>
<a href="https://aws.amazon.com/opensource/"><img src="https://github.com/apache/airflow/blob/main/providers/amazon/docs/integration-logos/AWS-Cloud-alt_light-bg@4x.png?raw=true" alt="AWS OpenSource" width="130px"></a>
```
Key improvements and SEO considerations:

*   **Clear Title and Description:** The title uses the target keyword ("Apache Airflow") and includes a benefit ("Orchestrate Your Workflows with Ease").  The first sentence acts as a concise, SEO-friendly description.
*   **Targeted Keywords:** The text naturally incorporates relevant keywords: "workflows," "data pipelines," "scheduling," "monitoring," "open-source."
*   **Headings and Structure:** Uses clear headings (H2) to organize content, improving readability and SEO.
*   **Bulleted Lists:** Key features are presented in a bulleted list, making them easy to scan and understand.
*   **Strong Call to Action:** Links to "Explore the official Apache Airflow repository" and "Dive into Airflow..." encourage engagement.
*   **Concise Language:** The language is direct and avoids unnecessary jargon.
*   **Emphasis on Benefits:** The description focuses on what users *get* from Airflow (manageability, scalability).
*   **Internal Linking:** Added internal linking to specific pages inside the same document.
*   **Clear Focus:** Maintains the original's purpose, with improved organization and style.
*   **Updated Version Life Cycle Table** Ensures up-to-date information about the available versions
*   **Sponsors information:** The information about the sponsors have been re-added in the end, making them also visible from the `pypi` view.