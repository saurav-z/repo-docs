# Apache Airflow: Orchestrate Your Workflows as Code

**Automate, schedule, and monitor your data pipelines with Apache Airflow, a powerful, open-source workflow management platform.**  [Explore the Apache Airflow project](https://github.com/apache/airflow).

[![License](https://img.shields.io/:license-Apache%202-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0.txt)
[![PyPI version](https://badge.fury.io/py/apache-airflow.svg)](https://badge.fury.io/py/apache-airflow)
[![Docker Pulls](https://img.shields.io/docker/pulls/apache/airflow.svg)](https://hub.docker.com/r/apache/airflow)
[![Contributors](https://img.shields.io/github/contributors/apache/airflow)](https://github.com/apache/airflow/graphs/contributors)

## Key Features

*   **Workflow as Code:** Define workflows (DAGs) using Python, making them maintainable, versionable, testable, and collaborative.
*   **Dynamic Pipelines:** Leverage Python's flexibility to generate pipelines dynamically and parameterize them.
*   **Rich UI:** Monitor pipeline progress, visualize dependencies, and troubleshoot issues through a user-friendly interface.
*   **Extensible Architecture:**  Extend Airflow's functionality with a wide range of operators and integrations or build your own.
*   **Scalable Scheduling:** The Airflow scheduler efficiently executes tasks across a distributed worker environment.

## Project Focus

Airflow excels at managing workflows that are mostly static and that change slowly. Airflow is commonly used for data processing but encourages idempotent tasks and leverages external services for high-volume, data-intensive workloads.

## Principles

*   **Dynamic**: Pipelines are defined in code, enabling dynamic dag generation and parameterization.
*   **Extensible**: The Airflow framework includes a wide range of built-in operators and can be extended to fit your needs.
*   **Flexible**: Airflow leverages the [**Jinja**](https://jinja.palletsprojects.com) templating engine, allowing rich customizations.

## Requirements

*   **Python:** 3.9, 3.10, 3.11, 3.12
*   **Kubernetes:** 1.30, 1.31, 1.32, 1.33
*   **Databases:** PostgreSQL, MySQL, SQLite (for development only)

[View full requirements](https://airflow.apache.org/docs/apache-airflow/stable/installation/)

## Installation

Install Apache Airflow using `pip` with a constraint file for repeatable installations. Comprehensive installation instructions can be found in the [INSTALLING.md](INSTALLING.md) file.

```bash
pip install 'apache-airflow==3.0.4' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.4/constraints-3.10.txt"
```

## User Interface Highlights

Airflow offers a rich UI for managing and monitoring your workflows:

*   **DAGs:** Overview of all DAGs.
*   **Assets:** Overview of Assets with dependencies.
*   **Grid:** Grid representation of a DAG that spans across time.
*   **Graph:** Visualizes DAG dependencies and their status for specific runs.
*   **Home:** Summary statistics of your Airflow environment.
*   **Backfill:** Backfilling a DAG for a specific date range.
*   **Code:** View source code of a DAG.

## Learn More

*   [Official Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
*   [Community Information](https://airflow.apache.org/community/)
*   [Airflow Improvement Proposals (AIPs)](https://cwiki.apache.org/confluence/display/AIRFLOW/Airflow+Improvement+Proposals)

## Contributing

Contribute to the project by following the [contributors' guide](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst).

## Sponsors

CI infrastructure sponsored by:

<a href="https://astronomer.io"><img src="https://assets2.astronomer.io/logos/logoForLIGHTbackground.png" alt="astronomer.io" width="250px"></a>
<a href="https://aws.amazon.com/opensource/"><img src="https://github.com/apache/airflow/blob/main/providers/amazon/docs/integration-logos/AWS-Cloud-alt_light-bg@4x.png?raw=true" alt="AWS OpenSource" width="130px"></a>