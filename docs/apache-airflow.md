# Apache Airflow: Automate, Schedule, and Monitor Workflows

**Apache Airflow is an open-source platform that allows you to programmatically author, schedule, and monitor complex workflows as directed acyclic graphs (DAGs).**  It empowers data engineers and data scientists to build robust, scalable, and maintainable data pipelines.  Find the original project at [https://github.com/apache/airflow](https://github.com/apache/airflow).

Key Features:

*   **Dynamic Workflow Definition:** Define pipelines in Python code, enabling dynamic DAG generation and parameterization.
*   **Extensible Architecture:** Leverage a wide array of built-in operators and customize the framework to fit your needs.
*   **Flexible Templating:** Utilize Jinja templating for rich customizations and workflow control.
*   **Scalable Scheduling and Execution:** The Airflow scheduler manages task execution across a distributed worker pool.
*   **Robust Monitoring and Visualization:**  The UI provides a clear view of pipeline progress, and helps troubleshoot issues.
*   **Idempotent Tasks & XComs:**  Airflow promotes idempotent tasks and uses XComs for passing metadata efficiently.

## Project Overview

Airflow is designed for workflows that benefit from static and slowly changing structures. It is commonly used for data processing but works well for a variety of tasks.

## Requirements

Apache Airflow is tested with the following:

|            | Main version (dev)     | Stable version (3.0.6) |
|------------|------------------------|------------------------|
| Python     | 3.10, 3.11, 3.12, 3.13 | 3.9, 3.10, 3.11, 3.12  |
| Platform   | AMD64/ARM64(\*)        | AMD64/ARM64(\*)        |
| Kubernetes | 1.30, 1.31, 1.32, 1.33 | 1.30, 1.31, 1.32, 1.33 |
| PostgreSQL | 13, 14, 15, 16, 17     | 13, 14, 15, 16, 17     |
| MySQL      | 8.0, 8.4, Innovation   | 8.0, 8.4, Innovation   |
| SQLite     | 3.15.0+                | 3.15.0+                |

\* Experimental

**Note**: SQLite should not be used in production; use the latest stable version for local development.

## Installation

For detailed installation instructions and setup for your local development environment, please refer to the [INSTALLING.md](INSTALLING.md) file.

## Installing from PyPI

Install Apache Airflow using pip with constraints for repeatable installs:

1.  Install Airflow:

    ```bash
    pip install 'apache-airflow==3.0.6' \
     --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.6/constraints-3.10.txt"
    ```

2.  Install with extras (e.g., postgres, google):

    ```bash
    pip install 'apache-airflow[postgres,google]==3.0.6' \
     --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.6/constraints-3.10.txt"
    ```

## User Interface

*   **DAGs:** Overview of all DAGs in your environment.
    ![DAGs](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/dags.png)
*   **Assets**: Overview of Assets with dependencies.
    ![Asset Dependencies](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/assets_graph.png)
*   **Grid:** Grid representation of a DAG that spans across time.
    ![Grid](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/grid.png)
*   **Graph:** Visualization of a DAG's dependencies and their current status for a specific run.
    ![Graph](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/graph.png)
*   **Home:** Summary statistics of your Airflow environment.
    ![Home](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/home.png)
*   **Backfill:** Backfilling a DAG for a specific date range.
    ![Backfill](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/backfill.png)
*   **Code:** Quick way to view source code of a DAG.
    ![Code](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/code.png)

## Additional Information

*   **[Official Source Code](https://downloads.apache.org/airflow):** Apache Airflow is an Apache Software Foundation project, with releases following the ASF Release Policy.
*   **Convenience Packages:**  Install with PyPI, Docker Images, and GitHub tags.
*   **[Semantic Versioning](https://semver.org/):**  Airflow 2.0.0+ uses SemVer.
*   **[Version Life Cycle](#version-life-cycle):**  Tracked to ensure stability.
*   **Support for Python and Kubernetes versions:**  Detailed in the [README](https://github.com/apache/airflow).
*   **[Contributing](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst):**  Contribute to Airflow by following the guidelines.

## Links

*   [Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
*   [Chat](https://s.apache.org/airflow-slack)
*   [Community Information](https://airflow.apache.org/community/)

## Sponsors

<!-- Ordered by most recently "funded" -->

<a href="https://astronomer.io"><img src="https://assets2.astronomer.io/logos/logoForLIGHTbackground.png" alt="astronomer.io" width="250px"></a>
<a href="https://aws.amazon.com/opensource/"><img src="https://github.com/apache/airflow/blob/main/providers/amazon/docs/integration-logos/AWS-Cloud-alt_light-bg@4x.png?raw=true" alt="AWS OpenSource" width="130px"></a>