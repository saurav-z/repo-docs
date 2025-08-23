# Apache Airflow: Orchestrate and Schedule Workflows with Code

**Apache Airflow** is a powerful platform for programmatically authoring, scheduling, and monitoring workflows, transforming complex data pipelines into manageable and collaborative projects.  [Explore the official Apache Airflow repository](https://github.com/apache/airflow) for more details.

[![License](https://img.shields.io/:license-Apache%202-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0.txt)
[![PyPI version](https://badge.fury.io/py/apache-airflow.svg)](https://badge.fury.io/py/apache-airflow)
[![Docker Pulls](https://img.shields.io/docker/pulls/apache/airflow.svg)](https://hub.docker.com/r/apache/airflow)
[![GitHub Build main](https://github.com/apache/airflow/actions/workflows/ci-amd.yml/badge.svg)](https://github.com/apache/airflow/actions)

<picture width="500">
  <img
    src="https://github.com/apache/airflow/blob/19ebcac2395ef9a6b6ded3a2faa29dc960c1e635/docs/apache-airflow/img/logos/wordmark_1.png?raw=true"
    alt="Apache Airflow logo"
  />
</picture>

Key features of Apache Airflow:

*   **Dynamic Workflows:** Define pipelines as code for dynamic DAG generation and parameterization.
*   **Extensible Architecture:** Benefit from a wide range of built-in operators and extend functionality as needed.
*   **Flexible Customization:** Leverage the Jinja templating engine for extensive pipeline customization.
*   **User-Friendly Interface:** Monitor progress, troubleshoot issues, and visualize pipelines through a rich user interface.
*   **Scalable Scheduling:**  Airflow scheduler executes your tasks on an array of workers.
*   **Idempotent Tasks:** Recommended for idempotent tasks which ensures that results of the task will be the same, and will not create duplicated data in a destination system.

## Core Principles

*   **Dynamic:** Define pipelines in code, enabling dynamic DAG generation and parameterization.
*   **Extensible:** The Airflow framework includes a wide range of built-in operators and can be extended to fit your needs.
*   **Flexible:** Airflow leverages the [**Jinja**](https://jinja.palletsprojects.com) templating engine, allowing rich customizations.

## Key Features

*   **DAGs:** Overview of all DAGs in your environment.
*   **Assets:** Overview of Assets with dependencies.
*   **Grid:** Grid representation of a DAG that spans across time.
*   **Graph:** Visualization of a DAG's dependencies and their current status for a specific run.
*   **Home:** Summary statistics of your Airflow environment.
*   **Backfill:** Backfilling a DAG for a specific date range.
*   **Code:** Quick way to view source code of a DAG.

## Requirements

Apache Airflow is tested with:

|            | Main version (dev)     | Stable version (3.0.5) |
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

Explore the official documentation for [installation](https://airflow.apache.org/docs/apache-airflow/stable/installation/), [getting started](https://airflow.apache.org/docs/apache-airflow/stable/start.html), and comprehensive tutorials.

> Note: If you're looking for documentation for the main branch (latest development branch): you can find it on [s.apache.org/airflow-docs](https://s.apache.org/airflow-docs/).

## Installing from PyPI

Install Airflow from PyPI with the `apache-airflow` package. Repeatable installation is ensured through constraint files.

### Installing Airflow:

```bash
pip install 'apache-airflow==3.0.5' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.5/constraints-3.10.txt"
```

### Installing with extras:

```bash
pip install 'apache-airflow[postgres,google]==3.0.5' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.5/constraints-3.10.txt"
```

## Installation

For detailed setup and installation instructions, see the [INSTALLING.md](INSTALLING.md) file.

## Official Source Code

Apache Airflow is an Apache Software Foundation (ASF) project with source code releases that adhere to the [ASF Release Policy](https://www.apache.org/legal/release-policy.html).

## Convenience Packages

Convenience methods for installing and using Airflow include:

*   [PyPI releases](https://pypi.org/project/apache-airflow/)
*   [Docker Images](https://hub.docker.com/r/apache/airflow)
*   [Tags in GitHub](https://github.com/apache/airflow/tags)

## Semantic Versioning

Airflow 2.0.0 and later follow a strict [SemVer](https://semver.org/) approach for releases, ensuring predictable versioning.

## Version Life Cycle

Apache Airflow version life cycle:

<!-- This table is automatically updated by prek scripts/ci/prek/supported_versions.py -->
<!-- Beginning of auto-generated table -->

| Version   | Current Patch/Minor   | State     | First Release   | Limited Maintenance   | EOL/Terminated   |
|-----------|-----------------------|-----------|-----------------|-----------------------|------------------|
| 3         | 3.0.5                 | Supported | Apr 22, 2025    | TBD                   | TBD              |
| 2         | 2.11.0                | Supported | Dec 17, 2020    | Oct 22, 2025          | Apr 22, 2026     |
| 1.10      | 1.10.15               | EOL       | Aug 27, 2018    | Dec 17, 2020          | June 17, 2021    |
| 1.9       | 1.9.0                 | EOL       | Jan 03, 2018    | Aug 27, 2018          | Aug 27, 2018     |
| 1.8       | 1.8.2                 | EOL       | Mar 19, 2017    | Jan 03, 2018          | Jan 03, 2018     |
| 1.7       | 1.7.1.2               | EOL       | Mar 28, 2016    | Mar 19, 2017          | Mar 19, 2017     |

<!-- End of auto-generated table -->

We highly recommend upgrading to the latest Airflow major release before the EOL date.

## Support for Python and Kubernetes versions

Airflow follows specific rules for Python and Kubernetes support, based on official release schedules.

## Base OS support for reference Airflow images

Airflow's container images are built using the stable Debian OS.

## Approach to dependencies of Airflow

Airflow uses a dependency management approach that balances application stability with flexibility for DAG developers.

## Contributing

Contribute to Apache Airflow by reviewing the [contributors' guide](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst).

## Voting Policy

*   Commits need a +1 vote from a committer who is not the author
*   When we do AIP voting, both PMC member's and committer's `+1s` are considered a binding vote.

## Who uses Apache Airflow?

Apache Airflow is used by approximately 500 organizations, find them [in the wild](https://github.com/apache/airflow/blob/main/INTHEWILD.md).

## Who maintains Apache Airflow?

The Apache Airflow [community](https://github.com/apache/airflow/graphs/contributors) and [core committers/maintainers](https://people.apache.org/committers-by-project.html#airflow) are responsible for reviewing and merging PRs and steering new feature requests.

## What goes into the next release?

The process of identifying what goes into each release and more details on the [What goes into the next release](dev/WHAT_GOES_INTO_THE_NEXT_RELEASE.md) document.

## Can I use the Apache Airflow logo in my presentation?

Yes, follow the Apache Foundation [trademark policies](https://www.apache.org/foundation/marks/#books) and the Apache Airflow [Brandbook](https://cwiki.apache.org/confluence/display/AIRFLOW/Brandbook).

## Links

*   [Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
*   [Chat](https://s.apache.org/airflow-slack)
*   [Community Information](https://airflow.apache.org/community/)

## Sponsors

The CI infrastructure for Apache Airflow has been sponsored by:

<!-- Ordered by most recently "funded" -->

<a href="https://astronomer.io"><img src="https://assets2.astronomer.io/logos/logoForLIGHTbackground.png" alt="astronomer.io" width="250px"></a>
<a href="https://aws.amazon.com/opensource/"><img src="https://github.com/apache/airflow/blob/main/providers/amazon/docs/integration-logos/AWS-Cloud-alt_light-bg@4x.png?raw=true" alt="AWS OpenSource" width="130px"></a>