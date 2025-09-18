# Apache Airflow: Orchestrate Complex Workflows with Code

**Apache Airflow is a leading open-source platform that empowers you to programmatically author, schedule, and monitor your workflows, turning complex data pipelines into manageable and efficient systems.**

[![License](https://img.shields.io/:license-Apache%202-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0.txt)
[![PyPI version](https://badge.fury.io/py/apache-airflow.svg)](https://badge.fury.io/py/apache-airflow)
[![Docker Pulls](https://img.shields.io/docker/pulls/apache/airflow.svg)](https://hub.docker.com/r/apache/airflow)
[![Contributors](https://img.shields.io/github/contributors/apache/airflow)](https://github.com/apache/airflow/graphs/contributors)

[Apache Airflow](https://github.com/apache/airflow) is a robust and scalable platform for orchestrating and managing data pipelines. Define workflows as code for maintainability, versioning, and collaboration.

**Key Features:**

*   **Dynamic Pipelines:** Define workflows in code, enabling dynamic DAG generation and parameterization.
*   **Extensible:** Includes a wide range of built-in operators and can be extended to fit your needs.
*   **Flexible:** Leverages Jinja templating for rich customizations.
*   **User-Friendly UI:** Visualize pipelines, monitor progress, and troubleshoot issues effectively.
*   **Scalable:** Designed to handle complex and data-intensive workflows.

**Get Started:**

*   **Requirements:**
    *   Python: 3.9, 3.10, 3.11, 3.12 (stable) or 3.10, 3.11, 3.12, 3.13 (dev)
    *   Kubernetes: 1.30, 1.31, 1.32, 1.33
    *   PostgreSQL, MySQL, SQLite.
*   **Installation:**

    ```bash
    pip install 'apache-airflow==3.0.6' \
     --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.6/constraints-3.10.txt"
    ```
    *(See detailed installation instructions in the [INSTALLING.md](INSTALLING.md) file or use the official documentation)*
*   **Explore the UI:**
    *   DAGs: Overview of all DAGs in your environment.
        ![DAGs](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/dags.png)
    *   Asset Dependencies: Overview of Assets with dependencies.
        ![Asset Dependencies](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/assets_graph.png)
    *   Grid: Grid representation of a DAG that spans across time.
        ![Grid](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/grid.png)
    *   Graph: Visualization of a DAG's dependencies and their current status for a specific run.
        ![Graph](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/graph.png)
    *   Home: Summary statistics of your Airflow environment.
        ![Home](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/home.png)
    *   Backfill: Backfilling a DAG for a specific date range.
        ![Backfill](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/backfill.png)
    *   Code: Quick way to view source code of a DAG.
        ![Code](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/code.png)

**Learn More:**

*   [Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
*   [Getting Started](https://airflow.apache.org/docs/apache-airflow/stable/start.html)
*   [Community Information](https://airflow.apache.org/community/)
*   [Contribution Guide](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst)

```

<!-- END Apache Airflow, please keep comment here to allow auto update of PyPI readme.md -->
```

**Additional Information:**

*   **Project Focus:** Airflow excels at static, slowly changing workflows, ideal for data processing and ETL pipelines.
*   **Contributing:**  Get involved by following the [contributor's guide](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst).
*   **Who Uses Airflow:**  Discover the [organizations using Airflow](https://github.com/apache/airflow/blob/main/INTHEWILD.md).
*   **Sponsors:**  Special thanks to [Astronomer.io](https://astronomer.io) and [AWS Open Source](https://aws.amazon.com/opensource/) for sponsoring the CI infrastructure.