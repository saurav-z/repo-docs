# Apache Airflow: Orchestrate Your Workflows as Code

**Automate, schedule, and monitor your workflows with Apache Airflow, a platform that empowers you to define pipelines as code, making them more manageable, versionable, and collaborative.**  Visit the [original repo](https://github.com/apache/airflow) for more details.

## Key Features

*   **Dynamic Workflows:** Define pipelines with code, enabling dynamic DAG generation and parameterization.
*   **Extensible Architecture:** Utilize a wide range of built-in operators and extend Airflow to fit your specific needs.
*   **Flexible Templating:** Leverage the power of Jinja templating for rich customizations.
*   **Intuitive User Interface:** Visualize pipeline progress, monitor tasks, and troubleshoot issues easily.
*   **Scalable Scheduling:** Execute tasks on a distributed array of workers with a robust scheduler.
*   **Integration Ecosystem:** Extensive integration with numerous services and tools.

## Project Overview

Apache Airflow is a leading platform for orchestrating complex workflows.  It allows you to define workflows (DAGs) as code, making them easier to maintain, version, and collaborate on.  Airflow’s scheduler executes your tasks on a fleet of workers, respecting the dependencies you specify. A rich user interface provides powerful monitoring and troubleshooting capabilities.

## Requirements

Apache Airflow is tested with:

|            | Main version (dev)     | Stable version (3.0.3) |
|------------|------------------------|------------------------|
| Python     | 3.10, 3.11, 3.12, 3.13 | 3.9, 3.10, 3.11, 3.12  |
| Platform   | AMD64/ARM64(\*)        | AMD64/ARM64(\*)        |
| Kubernetes | 1.30, 1.31, 1.32, 1.33 | 1.30, 1.31, 1.32, 1.33 |
| PostgreSQL | 13, 14, 15, 16, 17     | 13, 14, 15, 16, 17     |
| MySQL      | 8.0, 8.4, Innovation   | 8.0, 8.4, Innovation   |
| SQLite     | 3.15.0+                | 3.15.0+                |

\* Experimental

**Note**: MariaDB is not tested/recommended.

**Note**: SQLite is used in Airflow tests. Do not use it in production. We recommend
using the latest stable version of SQLite for local development.

**Note**: Airflow currently can be run on POSIX-compliant Operating Systems. For development, it is regularly
tested on fairly modern Linux Distros and recent versions of macOS.
On Windows you can run it via WSL2 (Windows Subsystem for Linux 2) or via Linux Containers.
The work to add Windows support is tracked via [#10388](https://github.com/apache/airflow/issues/10388), but
it is not a high priority. You should only use Linux-based distros as "Production" execution environment
as this is the only environment that is supported. The only distro that is used in our CI tests and that
is used in the [Community managed DockerHub image](https://hub.docker.com/p/apache/airflow) is
`Debian Bookworm`.

## Getting Started

Explore the official Apache Airflow documentation for detailed installation instructions, quick start guides, and comprehensive tutorials to get you up and running quickly.

## Installation

For setup instructions, please refer to the [INSTALLING.md](INSTALLING.md) file.

## Installing from PyPI

Install Airflow from PyPI using `pip`. Install with constraints to ensure a working install:

```bash
pip install 'apache-airflow==3.0.3' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.3/constraints-3.10.txt"
```

Install with extras, such as `postgres` and `google`:

```bash
pip install 'apache-airflow[postgres,google]==3.0.3' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.3/constraints-3.10.txt"
```

## User Interface

Airflow's intuitive UI provides a clear view of your workflows:

*   **DAGs:** Overview of all DAGs.
*   **Assets**: Overview of Assets with dependencies.
*   **Grid:** Time-based representation of DAGs.
*   **Graph:** Visual representation of dependencies and status.
*   **Home:** Summary statistics.
*   **Backfill:** Backfilling a DAG for a specific date range.
*   **Code:** View DAG source code.

## Contributing

Contribute to Apache Airflow by following the [contributors' guide](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst) and [quickstart](https://github.com/apache/airflow/blob/main/contributing-docs/03_contributors_quick_start.rst).

## Links

*   [Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
*   [Chat](https://s.apache.org/airflow-slack)
*   [Community Information](https://airflow.apache.org/community/)

## Sponsors

<!-- Ordered by most recently "funded" -->

<a href="https://astronomer.io"><img src="https://assets2.astronomer.io/logos/logoForLIGHTbackground.png" alt="astronomer.io" width="250px"></a>
<a href="https://aws.amazon.com/opensource/"><img src="https://github.com/apache/airflow/blob/main/providers/amazon/docs/integration-logos/AWS-Cloud-alt_light-bg@4x.png?raw=true" alt="AWS OpenSource" width="130px"></a>