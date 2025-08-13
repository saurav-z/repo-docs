# Apache Airflow: Orchestrate Your Workflows with Code

**Automate, schedule, and monitor your data pipelines with Apache Airflow, a platform that defines workflows as code.** [View the original repo](https://github.com/apache/airflow)

[![License](https://img.shields.io/:license-Apache%202-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0.txt)
[![PyPI version](https://badge.fury.io/py/apache-airflow.svg)](https://badge.fury.io/py/apache-airflow)
[![Docker Pulls](https://img.shields.io/docker/pulls/apache/airflow.svg)](https://hub.docker.com/r/apache/airflow)
[![Contributors](https://img.shields.io/github/contributors/apache/airflow)](https://github.com/apache/airflow/graphs/contributors)

Apache Airflow is a powerful platform for programmatically authoring, scheduling, and monitoring workflows.  It allows you to define workflows as code (DAGs), making them more maintainable, testable, and collaborative.

**Key Features:**

*   **Dynamic:** Define pipelines in code for dynamic DAG generation and parameterization.
*   **Extensible:** Leverage a wide array of built-in operators and easily extend the framework.
*   **Flexible:** Customize your pipelines with the Jinja templating engine.
*   **Monitoring & Management:** Rich UI to visualize, monitor, and troubleshoot workflows in production.
*   **Scalable:** Airflow executes tasks on a fleet of workers.

## Installation and Getting Started

Comprehensive installation guides are available, see [INSTALLING.md](INSTALLING.md). For additional help, visit the official Airflow website for the [stable release documentation](https://airflow.apache.org/docs/apache-airflow/stable/installation/), [getting started guide](https://airflow.apache.org/docs/apache-airflow/stable/start.html), or detailed [tutorial](https://airflow.apache.org/docs/apache-airflow/stable/tutorial/).

## Benefits

*   **Improved Maintainability:** Workflows defined as code are easier to understand and update.
*   **Enhanced Collaboration:** Version control and testing become natural parts of your workflow management.
*   **Increased Reliability:** Automated scheduling and monitoring ensure smooth pipeline execution.

## Installing from PyPI

Install Apache Airflow using `pip` with constraint files for repeatable installations.

**1. Install Airflow:**

```bash
pip install 'apache-airflow==3.0.4' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.4/constraints-3.10.txt"
```

**2. Install with extras (e.g., postgres, google):**

```bash
pip install 'apache-airflow[postgres,google]==3.0.4' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.4/constraints-3.10.txt"
```

## Key Details

### Requirements

Apache Airflow is tested with the following:

|            | Main version (dev)     | Stable version (3.0.4) |
|------------|------------------------|------------------------|
| Python     | 3.10, 3.11, 3.12, 3.13 | 3.9, 3.10, 3.11, 3.12  |
| Platform   | AMD64/ARM64(\*)        | AMD64/ARM64(\*)        |
| Kubernetes | 1.30, 1.31, 1.32, 1.33 | 1.30, 1.31, 1.32, 1.33 |
| PostgreSQL | 13, 14, 15, 16, 17     | 13, 14, 15, 16, 17     |
| MySQL      | 8.0, 8.4, Innovation   | 8.0, 8.4, Innovation   |
| SQLite     | 3.15.0+                | 3.15.0+                |

\* Experimental

### UI Overview
*   **DAGs**: Overview of all DAGs in your environment.

    ![DAGs](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/dags.png)

*   **Assets**: Overview of Assets with dependencies.

    ![Asset Dependencies](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/assets_graph.png)

*   **Grid**: Grid representation of a DAG that spans across time.

    ![Grid](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/grid.png)

*   **Graph**: Visualization of a DAG's dependencies and their current status for a specific run.

    ![Graph](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/graph.png)

*   **Home**: Summary statistics of your Airflow environment.

    ![Home](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/home.png)

*   **Backfill**: Backfilling a DAG for a specific date range.

    ![Backfill](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/backfill.png)

*   **Code**: Quick way to view source code of a DAG.

    ![Code](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/code.png)

### Semantic Versioning

Airflow uses a strict SemVer approach, which is detailed under the [Semantic versioning](#semantic-versioning) section.

### Version Life Cycle

See the table under [Version Life Cycle](#version-life-cycle) for specific version support details.

### Base OS and Dependencies
For details on base OS support, refer to the [Base OS support for reference Airflow images](#base-os-support-for-reference-airflow-images). For information on how Airflow handles dependencies, check out the [Approach to dependencies of Airflow](#approach-to-dependencies-of-airflow) section.

## Community and Contribution

### Contributing

Contribute to Apache Airflow by following the guidelines in the [contributors' guide](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst).

### Who Uses Apache Airflow?

A list of organizations using Airflow is found [in the wild](https://github.com/apache/airflow/blob/main/INTHEWILD.md).

### Who Maintains Apache Airflow?

Airflow is maintained by a vibrant community and the [core committers/maintainers](https://people.apache.org/committers-by-project.html#airflow).

### Voting Policy

* Commits need a +1 vote from a committer who is not the author
* When we do AIP voting, both PMC member's and committer's `+1s` are considered a binding vote.

## Resources

*   [Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
*   [Chat](https://s.apache.org/airflow-slack)
*   [Community Information](https://airflow.apache.org/community/)

## Sponsors

[Sponsors](#sponsors)