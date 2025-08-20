# Apache Airflow: Automate, Schedule, and Monitor Workflows

**Apache Airflow** is a powerful platform for programmatically authoring, scheduling, and monitoring workflows.

[![License](https://img.shields.io/:license-Apache%202-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0.txt)
[![PyPI version](https://badge.fury.io/py/apache-airflow.svg)](https://badge.fury.io/py/apache-airflow)
[![Docker Pulls](https://img.shields.io/docker/pulls/apache/airflow.svg)](https://hub.docker.com/r/apache/airflow)
[![Contributors](https://img.shields.io/github/contributors/apache/airflow)](https://github.com/apache/airflow/graphs/contributors)

**Streamline your data pipelines and task orchestration with Airflow, a flexible and scalable workflow management platform.  [Explore the original repository](https://github.com/apache/airflow).**

**Key Features:**

*   **Dynamic:** Define pipelines as code, enabling dynamic DAG generation and parameterization.
*   **Extensible:** Utilize a wide range of built-in operators and extend the platform to meet your specific needs.
*   **Flexible:** Leverage the **Jinja** templating engine for extensive customization.
*   **Scalable:** Easily handle complex workflows and growing data volumes.
*   **UI-Driven:** A rich user interface to visualize, monitor, and troubleshoot your pipelines.

<picture width="500">
  <img
    src="https://github.com/apache/airflow/blob/19ebcac2395ef9a6b6ded3a2faa29dc960c1e635/docs/apache-airflow/img/logos/wordmark_1.png?raw=true"
    alt="Apache Airflow logo"
  />
</picture>

## Project Overview

Apache Airflow is designed for creating and managing complex workflows, particularly in data processing and ETL pipelines. It offers a robust solution for automating tasks, scheduling their execution, and monitoring their progress. Airflow's core principles center around dynamic pipeline definition, extensibility, and flexibility, allowing users to customize and adapt the platform to diverse requirements.

## Principles

*   **Dynamic:** Pipelines are defined in code, enabling dynamic DAG generation and parameterization.
*   **Extensible:** The Airflow framework includes a wide range of built-in operators and can be extended to fit your needs.
*   **Flexible:** Airflow leverages the [**Jinja**](https://jinja.palletsprojects.com) templating engine, allowing rich customizations.

## Requirements

This section details the supported versions of key components.
**Current Testing and Support for Python and Kubernetes Versions**

|            | Main version (dev)     | Stable version (3.0.4) |
|------------|------------------------|------------------------|
| Python     | 3.10, 3.11, 3.12, 3.13 | 3.9, 3.10, 3.11, 3.12  |
| Platform   | AMD64/ARM64(\*)        | AMD64/ARM64(\*)        |
| Kubernetes | 1.30, 1.31, 1.32, 1.33 | 1.30, 1.31, 1.32, 1.33 |
| PostgreSQL | 13, 14, 15, 16, 17     | 13, 14, 15, 16, 17     |
| MySQL      | 8.0, 8.4, Innovation   | 8.0, 8.4, Innovation   |
| SQLite     | 3.15.0+                | 3.15.0+                |

\* Experimental

## Getting Started

Get up and running quickly! Visit the official Airflow documentation for installation, tutorials, and more:
[Airflow Official Documentation](https://airflow.apache.org/docs/apache-airflow/stable/).

## Installing from PyPI

Install Apache Airflow using `pip` with constraints for a repeatable setup.  Detailed installation instructions including extras can be found in the documentation.

```bash
pip install 'apache-airflow==3.0.4' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.4/constraints-3.10.txt"
```
## Installation
For comprehensive instructions on setting up your local development environment and installing Apache Airflow, please refer to the [INSTALLING.md](INSTALLING.md) file.

## Official Source Code

Apache Airflow is an [Apache Software Foundation](https://www.apache.org) (ASF) project.  Official releases adhere to the [ASF Release Policy](https://www.apache.org/legal/release-policy.html) and are available for download.

## Convenience Packages

Beyond official releases, Airflow offers convenience packages such as:

*   **PyPI Releases:** Install Airflow using `pip`.
*   **Docker Images:** Deploy Airflow using Docker for containerization. Explore [Docker Images](https://hub.docker.com/r/apache/airflow) for customization options.
*   **GitHub Tags:** Access the git project sources used for official source package generation via git.

## User Interface

The Airflow UI provides a rich set of tools to visualize, monitor, and manage your workflows:

*   **DAGs:** Overview of all DAGs.
*   **Assets**: Overview of Assets with dependencies.
*   **Grid:** Time-based DAG representation.
*   **Graph:** Visualize DAG dependencies and statuses.
*   **Home:** Summary environment statistics.
*   **Backfill:** Backfilling a DAG for a date range.
*   **Code:** View DAG source code.

(Include UI image examples here, such as the ones in the original README.  For example:)

![DAGs](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/dags.png)

## Semantic Versioning

Airflow adheres to [SemVer](https://semver.org/) for all released packages, ensuring predictable versioning for core Airflow and its components.

## Version Life Cycle

The release schedule follows standard processes, and is listed as follows:

<!-- This table is automatically updated by prek scripts/ci/prek/supported_versions.py -->
<!-- Beginning of auto-generated table -->

| Version   | Current Patch/Minor   | State     | First Release   | Limited Maintenance   | EOL/Terminated   |
|-----------|-----------------------|-----------|-----------------|-----------------------|------------------|
| 3         | 3.0.4                 | Supported | Apr 22, 2025    | TBD                   | TBD              |
| 2         | 2.11.0                | Supported | Dec 17, 2020    | Oct 22, 2025          | Apr 22, 2026     |
| 1.10      | 1.10.15               | EOL       | Aug 27, 2018    | Dec 17, 2020          | June 17, 2021    |
| 1.9       | 1.9.0                 | EOL       | Jan 03, 2018    | Aug 27, 2018          | Aug 27, 2018     |
| 1.8       | 1.8.2                 | EOL       | Mar 19, 2017    | Jan 03, 2018          | Jan 03, 2018     |
| 1.7       | 1.7.1.2               | EOL       | Mar 28, 2016    | Mar 19, 2017          | Mar 19, 2017     |

<!-- End of auto-generated table -->

## Support for Python and Kubernetes versions

As of Airflow 2.0, specific rules are followed for Python and Kubernetes support, aligned with their official release schedules.

## Base OS support for reference Airflow images

Airflow provides container images with a stable Debian OS, supporting various Python versions, database clients, and popular providers.

## Approach to dependencies of Airflow

Airflow utilizes a 'constraints' approach to manage dependencies, ensuring repeatable installations while allowing users to upgrade dependencies. The pyproject.toml file contains the core dependencies. The project does not upper-bound the dependencies by default, and it should only be done if the dependency is known to cause problems.

## Contributing

Contribute to Airflow! The [contributors' guide](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst) provides detailed instructions.  Quickstart [here](https://github.com/apache/airflow/blob/main/contributing-docs/03_contributors_quick_start.rst).

## Who uses Apache Airflow?

Airflow is used by a diverse set of organizations. See a list of users [in the wild](https://github.com/apache/airflow/blob/main/INTHEWILD.md).

## Who maintains Apache Airflow?

Airflow is maintained by a global community, with core committers responsible for code reviews and feature direction.

## Voting Policy

Airflow operates with a community-driven voting policy for commits and AIPs.

## What goes into the next release?

Learn about the release process and milestones in the [dev/WHAT_GOES_INTO_THE_NEXT_RELEASE.md](dev/WHAT_GOES_INTO_THE_NEXT_RELEASE.md) document.

## Can I use the Apache Airflow logo in my presentation?

Yes! Follow the Apache Foundation [trademark policies](https://www.apache.org/foundation/marks/#books) and the Apache Airflow [Brandbook](https://cwiki.apache.org/confluence/display/AIRFLOW/Brandbook).

## Links

*   [Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
*   [Chat](https://s.apache.org/airflow-slack)
*   [Community Information](https://airflow.apache.org/community/)

## Sponsors

(List and link to sponsors)
<a href="https://astronomer.io"><img src="https://assets2.astronomer.io/logos/logoForLIGHTbackground.png" alt="astronomer.io" width="250px"></a>
<a href="https://aws.amazon.com/opensource/"><img src="https://github.com/apache/airflow/blob/main/providers/amazon/docs/integration-logos/AWS-Cloud-alt_light-bg@4x.png?raw=true" alt="AWS OpenSource" width="130px"></a>