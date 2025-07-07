# Apache Airflow: Orchestrate Your Workflows with Code

**Apache Airflow** is an open-source platform that allows you to programmatically author, schedule, and monitor complex workflows.  Improve data pipelines with code, enhance maintainability, and streamline collaboration, using the original repo [here](https://github.com/apache/airflow).

**Key Features:**

*   **Dynamic:** Define pipelines in code for flexibility and parameterization.
*   **Extensible:** Use a wide array of built-in operators and customize to fit your needs.
*   **Flexible:** Leverage the Jinja templating engine for rich customizations.
*   **Scalable**: Built for parallel and distributed execution of tasks.
*   **User-Friendly UI**: Visualize, monitor, and troubleshoot workflows with ease.

**Key Advantages:**

*   **Maintainability**: Code-based workflows are easier to understand, test, and version.
*   **Scalability**: Handle complex workflows with efficient task scheduling and execution.
*   **Collaboration**: Share and collaborate on workflows with version control and code reviews.

<picture width="500">
  <img
    src="https://github.com/apache/airflow/blob/19ebcac2395ef9a6b6ded3a2faa29dc960c1e635/docs/apache-airflow/img/logos/wordmark_1.png?raw=true"
    alt="Apache Airflow logo"
  />
</picture>

## Getting Started

*   **Installation**: Comprehensive instructions are available in the [INSTALLING.md](INSTALLING.md) file.
*   **Documentation**: Refer to the official Apache Airflow documentation for installation, getting started guides, and tutorials.
*   **Community**: Join the conversation via the Airflow Slack channel and the [Community Information](https://airflow.apache.org/community/) page.

## Installing from PyPI

Install Apache Airflow with `pip`:

```bash
pip install 'apache-airflow==3.0.2' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.2/constraints-3.10.txt"
```

Install with extras:

```bash
pip install 'apache-airflow[postgres,google]==3.0.2' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.2/constraints-3.10.txt"
```

## Requirements

Apache Airflow is tested with:

|            | Main version (dev)     | Stable version (3.0.2) |
|------------|------------------------|------------------------|
| Python     | 3.10, 3.11, 3.12       | 3.9, 3.10, 3.11, 3.12  |
| Platform   | AMD64/ARM64(\*)        | AMD64/ARM64(\*)        |
| Kubernetes | 1.30, 1.31, 1.32, 1.33 | 1.30, 1.31, 1.32, 1.33 |
| PostgreSQL | 13, 14, 15, 16, 17     | 13, 14, 15, 16, 17     |
| MySQL      | 8.0, 8.4, Innovation   | 8.0, 8.4, Innovation   |
| SQLite     | 3.15.0+                | 3.15.0+                |

\* Experimental

## User Interface

Airflow's user-friendly interface offers powerful features:

*   **DAGs**: Overview of all DAGs
*   **Assets**: Overview of Assets with dependencies.
*   **Grid**: Grid representation of a DAG that spans across time.
*   **Graph**: Visualize a DAG's dependencies and current status.
*   **Home**: Summary statistics of your Airflow environment.
*   **Backfill**: Backfilling a DAG for a specific date range.
*   **Code**: Quick way to view source code of a DAG.

## Key project details:
*   **Project Focus**: Airflow is best suited for static, slowly-changing workflows.
*   **Semantic Versioning**: Airflow follows SemVer for core packages.
*   **Version Life Cycle**:  See the README for the release life cycle.
*   **Support for Python and Kubernetes versions**:  See the README for supported versions.
*   **Base OS support for reference Airflow images**: See the README for base image support.
*   **Approach to dependencies of Airflow**: See the README for dependency management.

## Contribute

Join the Airflow community and help build the platform.  See the [contributors' guide](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst) for detailed instructions.

## Links

*   [Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
*   [Chat](https://s.apache.org/airflow-slack)
*   [Community Information](https://airflow.apache.org/community/)

## Sponsors

*   [Astronmer.io](https://astronomer.io)
*   [AWS OpenSource](https://aws.amazon.com/opensource/)