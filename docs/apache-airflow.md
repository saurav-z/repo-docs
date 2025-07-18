# Apache Airflow: The Open-Source Workflow Orchestration Platform

**Orchestrate your workflows with code, making them more maintainable, versionable, and collaborative with Apache Airflow!** Visit the [original repo](https://github.com/apache/airflow) for the source code.

## Key Features

*   **Dynamic**: Define pipelines in code, allowing for dynamic DAG generation and parameterization.
*   **Extensible**: Leverage a wide range of built-in operators and easily extend the framework to fit your needs.
*   **Flexible**: Utilize the **Jinja** templating engine for rich customizations.
*   **User-Friendly UI**: Visualize pipelines, monitor progress, and troubleshoot issues with an intuitive user interface.
*   **Scalable**: Execute tasks on a scalable array of workers.

## User Interface Overview
*   **DAGs**: Overview of all DAGs in your environment.
*   **Assets**: Overview of Assets with dependencies.
*   **Grid**: Grid representation of a DAG that spans across time.
*   **Graph**: Visualization of a DAG's dependencies and their current status for a specific run.
*   **Home**: Summary statistics of your Airflow environment.
*   **Backfill**: Backfilling a DAG for a specific date range.
*   **Code**: Quick way to view source code of a DAG.

## Getting Started

Explore the power of Airflow with the official documentation! Get help with [installing Airflow](https://airflow.apache.org/docs/apache-airflow/stable/installation/), [getting started](https://airflow.apache.org/docs/apache-airflow/stable/start.html), and completing the [tutorial](https://airflow.apache.org/docs/apache-airflow/stable/tutorial/).

## Key Benefits of Apache Airflow

*   **Version Control**: Define your workflows as code, enabling seamless version control and collaboration.
*   **Monitoring & Alerts**: Proactively monitor your workflows and receive alerts for any issues.
*   **Scalability**: Airflow's architecture allows you to scale your infrastructure to handle large and complex workflows.
*   **Extensibility**: Connect to any data source or service with a variety of built-in operators and custom plugins.

## Installation
Comprehensive instructions for setting up your local development environment and installing Apache Airflow can be found in the [INSTALLING.md](INSTALLING.md) file.

## Installing from PyPI

Install Apache Airflow from PyPI with repeatable installation using constraint files:

```bash
pip install 'apache-airflow==3.0.3' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.3/constraints-3.10.txt"
```

Install with extras (i.e., postgres, google):

```bash
pip install 'apache-airflow[postgres,google]==3.0.3' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.3/constraints-3.10.txt"
```

For information on installing provider distributions, check
[providers](http://airflow.apache.org/docs/apache-airflow-providers/index.html).

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

## Official Source Code

As an [Apache Software Foundation](https://www.apache.org) (ASF) project, Apache Airflow's official releases are:

*   Released under the [ASF Release Policy](https://www.apache.org/legal/release-policy.html)
*   Downloadable from [the ASF Distribution Directory](https://downloads.apache.org/airflow)
*   Cryptographically signed by the release manager

## Versioning
Airflow 2.0 and later use strict [SemVer](https://semver.org/). For details, see [Semantic versioning](#semantic-versioning).

## Version Life Cycle

Apache Airflow version life cycle:

<!-- This table is automatically updated by pre-commit scripts/ci/pre_commit/supported_versions.py -->
<!-- Beginning of auto-generated table -->

| Version   | Current Patch/Minor   | State     | First Release   | Limited Maintenance   | EOL/Terminated   |
|-----------|-----------------------|-----------|-----------------|-----------------------|------------------|
| 3         | 3.0.3                 | Supported | Apr 22, 2025    | TBD                   | TBD              |
| 2         | 2.11.0                | Supported | Dec 17, 2020    | Oct 22, 2025          | Apr 22, 2026     |
| 1.10      | 1.10.15               | EOL       | Aug 27, 2018    | Dec 17, 2020          | June 17, 2021    |
| 1.9       | 1.9.0                 | EOL       | Jan 03, 2018    | Aug 27, 2018          | Aug 27, 2018     |
| 1.8       | 1.8.2                 | EOL       | Mar 19, 2017    | Jan 03, 2018          | Jan 03, 2018     |
| 1.7       | 1.7.1.2               | EOL       | Mar 28, 2016    | Mar 19, 2017          | Mar 19, 2017     |

<!-- End of auto-generated table -->

## Contributing

Help build Apache Airflow! Find detailed instructions in our [contributors' guide](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst).

## Community

*   [Chat](https://s.apache.org/airflow-slack)
*   [Community Information](https://airflow.apache.org/community/)

## Sponsors
<!-- Ordered by most recently "funded" -->

<a href="https://astronomer.io"><img src="https://assets2.astronomer.io/logos/logoForLIGHTbackground.png" alt="astronomer.io" width="250px"></a>
<a href="https://aws.amazon.com/opensource/"><img src="https://github.com/apache/airflow/blob/main/providers/amazon/docs/integration-logos/AWS-Cloud-alt_light-bg@4x.png?raw=true" alt="AWS OpenSource" width="130px"></a>