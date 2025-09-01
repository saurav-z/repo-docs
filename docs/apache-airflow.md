# Apache Airflow: Orchestrate Your Workflows with Code

**Apache Airflow is a powerful, open-source platform that allows you to programmatically author, schedule, and monitor your data pipelines.**  [Explore the Apache Airflow Repository](https://github.com/apache/airflow)

[![License](https://img.shields.io/:license-Apache%202-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0.txt)
[![PyPI version](https://badge.fury.io/py/apache-airflow.svg)](https://badge.fury.io/py/apache-airflow)
[![Docker Pulls](https://img.shields.io/docker/pulls/apache/airflow.svg)](https://hub.docker.com/r/apache/airflow)
[![Contributors](https://img.shields.io/github/contributors/apache/airflow)](https://github.com/apache/airflow/graphs/contributors)
[![Slack Status](https://img.shields.io/badge/slack-join_chat-white.svg?logo=slack&style=social)](https://s.apache.org/airflow-slack)

**Key Features:**

*   **Code-Defined Workflows:** Define your data pipelines as code (DAGs) for maintainability, versioning, and collaboration.
*   **Dynamic DAG Generation:**  Airflow enables dynamic DAG generation and parameterization.
*   **Extensible Architecture:** Leverage a wide range of built-in operators and easily extend Airflow to fit your specific needs.
*   **Flexible Templating:** Utilize the Jinja templating engine for rich customization.
*   **Scheduling and Execution:** Airflow's scheduler executes tasks on a distributed worker pool based on defined dependencies.
*   **Rich UI:** Visualize pipelines, monitor progress, and troubleshoot issues through Airflow's intuitive user interface.
*   **Idempotent tasks:** Tasks should ideally be idempotent (i.e., results of the task will be the same, and will not create duplicated data in a destination system), and should not pass large quantities of data from one task to the next

## Why Choose Apache Airflow?

Airflow simplifies the complex task of workflow orchestration by:

*   **Increasing Maintainability:** Workflows defined in code are easier to understand, test, and modify.
*   **Improving Collaboration:** Code-based workflows promote teamwork and version control.
*   **Boosting Reliability:**  The scheduler and UI provide robust monitoring and troubleshooting capabilities.

##  Getting Started with Apache Airflow

Get started with Airflow by following these quick steps:

1.  **Installation:** Refer to the [INSTALLING.md](INSTALLING.md) file for detailed instructions to set up your local development environment.  Also, explore the [Installation](#installation) section.
2.  **Documentation:** Visit the official Apache Airflow [website](https://airflow.apache.org/docs/apache-airflow/stable/) for documentation and the [Getting Started](https://airflow.apache.org/docs/apache-airflow/stable/start.html) guide.

##  Installation from PyPI

Install Apache Airflow with the following command:

```bash
pip install 'apache-airflow==3.0.6' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.6/constraints-3.10.txt"
```

Install with extras:

```bash
pip install 'apache-airflow[postgres,google]==3.0.6' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.6/constraints-3.10.txt"
```

For further information, check out [Installing from PyPI](#installing-from-pypi).

##  Key Components and Concepts

*   **DAGs (Directed Acyclic Graphs):** Define workflows as code.
*   **Operators:** Pre-built tasks for common operations (e.g., running a Python script, executing a SQL query).
*   **Schedulers:** Automate the execution of your workflows.
*   **Workers:** Execute tasks in a distributed manner.
*   **UI (User Interface):** Monitor and manage your workflows.

##  User Interface

The Airflow UI provides:

*   **DAGs:** Overview of all DAGs.
*   **Assets:** Overview of Assets with dependencies.
*   **Grid:** Grid representation of a DAG that spans across time.
*   **Graph:** Visualization of DAG dependencies.
*   **Home:** Summary statistics.
*   **Backfill:** Backfilling a DAG.
*   **Code:** View DAG source code.

##  Semantic Versioning

Apache Airflow uses [SemVer](https://semver.org/) for versioning.  This provides a clear understanding of backwards compatibility and the impact of updates.  See the [Semantic versioning](#semantic-versioning) section for more details.

##  Version Life Cycle

Apache Airflow has a defined [version life cycle](#version-life-cycle), which includes support periods and EOL dates.

##  Requirements

Apache Airflow is tested with these configurations:

|            | Main version (dev)     | Stable version (3.0.6) |
|------------|------------------------|------------------------|
| Python     | 3.10, 3.11, 3.12, 3.13 | 3.9, 3.10, 3.11, 3.12  |
| Platform   | AMD64/ARM64(\*)        | AMD64/ARM64(\*)        |
| Kubernetes | 1.30, 1.31, 1.32, 1.33 | 1.30, 1.31, 1.32, 1.33 |
| PostgreSQL | 13, 14, 15, 16, 17     | 13, 14, 15, 16, 17     |
| MySQL      | 8.0, 8.4, Innovation   | 8.0, 8.4, Innovation   |
| SQLite     | 3.15.0+                | 3.15.0+                |

##  Contributing

Contribute to Apache Airflow by reviewing our [contributors' guide](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst) for a comprehensive overview and also, you can see the [contribution quickstart](https://github.com/apache/airflow/blob/main/contributing-docs/03_contributors_quick_start.rst).

##  Resources

*   [Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
*   [Chat](https://s.apache.org/airflow-slack)
*   [Community Information](https://airflow.apache.org/community/)

## Sponsors

<a href="https://astronomer.io"><img src="https://assets2.astronomer.io/logos/logoForLIGHTbackground.png" alt="astronomer.io" width="250px"></a>
<a href="https://aws.amazon.com/opensource/"><img src="https://github.com/apache/airflow/blob/main/providers/amazon/docs/integration-logos/AWS-Cloud-alt_light-bg@4x.png?raw=true" alt="AWS OpenSource" width="130px"></a>