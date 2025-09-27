# Apache Airflow: Orchestrate and Automate Your Workflows with Ease

**Simplify complex data pipelines and automate your workflows with Apache Airflow, the leading open-source platform for programmatic workflow management.**  [Explore the official Apache Airflow repository](https://github.com/apache/airflow).

[![License](https://img.shields.io/:license-Apache%202-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0.txt)
[![PyPI version](https://badge.fury.io/py/apache-airflow.svg)](https://badge.fury.io/py/apache-airflow)
[![Docker Pulls](https://img.shields.io/docker/pulls/apache/airflow.svg)](https://hub.docker.com/r/apache/airflow)
[![Contributors](https://img.shields.io/github/contributors/apache/airflow)](https://github.com/apache/airflow/graphs/contributors)

<picture width="500">
  <img
    src="https://github.com/apache/airflow/blob/19ebcac2395ef9a6b6ded3a2faa29dc960c1e635/docs/apache-airflow/img/logos/wordmark_1.png?raw=true"
    alt="Apache Airflow logo"
  />
</picture>

## Key Features

*   **Dynamic Workflows:** Define pipelines as code, allowing for dynamic DAG generation and parameterization.
*   **Extensibility:** Leverage a rich set of built-in operators and easily extend the framework to meet your specific needs.
*   **Flexibility:** Utilize the **Jinja** templating engine for rich customization of your workflows.
*   **Scalability:** Executes tasks on a distributed array of workers, designed to handle complex workloads.
*   **User-Friendly Interface:** Monitor and troubleshoot your pipelines through a powerful and intuitive web UI.

## Project Focus

Airflow is designed for workflows that are mostly static and change slowly over time. The platform focuses on the clear definition of units of work and task continuity.

## Principles

*   **Dynamic**: Pipelines are defined in code, enabling dynamic dag generation and parameterization.
*   **Extensible**: The Airflow framework includes a wide range of built-in operators and can be extended to fit your needs.
*   **Flexible**: Airflow leverages the [**Jinja**](https://jinja.palletsprojects.com) templating engine, allowing rich customizations.

## Requirements

Airflow is tested with:

|            | Main version (dev)           | Stable version (3.1.0) |
|------------|------------------------------|------------------------|
| Python     | 3.10, 3.11, 3.12, 3.13       | 3.9, 3.10, 3.11, 3.12  |
| Platform   | AMD64/ARM64(\*)              | AMD64/ARM64(\*)        |
| Kubernetes | 1.30, 1.31, 1.32, 1.33, 1.34 | 1.30, 1.31, 1.32, 1.33 |
| PostgreSQL | 13, 14, 15, 16, 17           | 13, 14, 15, 16, 17     |
| MySQL      | 8.0, 8.4, Innovation         | 8.0, 8.4, Innovation   |
| SQLite     | 3.15.0+                      | 3.15.0+                |

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

Explore the official documentation for installation, getting started guides, and tutorials.

## Installing from PyPI

Install Apache Airflow using pip and constraint files for repeatable installations.

## Installation

For comprehensive instructions on setting up your local development environment and installing Apache Airflow, please refer to the [INSTALLING.md](INSTALLING.md) file.

## Official Source Code

Apache Airflow is an Apache Software Foundation (ASF) project and our official source code releases:

*   Follow the [ASF Release Policy](https://www.apache.org/legal/release-policy.html)
*   Can be downloaded from [the ASF Distribution Directory](https://downloads.apache.org/airflow)
*   Are cryptographically signed by the release manager
*   Are officially voted on by the PMC members during the
    [Release Approval Process](https://www.apache.org/legal/release-policy.html#release-approval)

Following the ASF rules, the source packages released must be sufficient for a user to build and test the
release provided they have access to the appropriate platform and tools.

## Convenience packages

There are other ways of installing and using Airflow. Those are "convenience" methods - they are
not "official releases" as stated by the `ASF Release Policy`, but they can be used by the users
who do not want to build the software themselves.

Those are - in the order of most common ways people install Airflow:

-   [PyPI releases](https://pypi.org/project/apache-airflow/) to install Airflow using standard `pip` tool
-   [Docker Images](https://hub.docker.com/r/apache/airflow) to install airflow via
    `docker` tool, use them in Kubernetes, Helm Charts, `docker-compose`, `docker swarm`, etc. You can
    read more about using, customizing, and extending the images in the
    [Latest docs](https://airflow.apache.org/docs/docker-stack/index.html), and
    learn details on the internals in the [images](https://airflow.apache.org/docs/docker-stack/index.html) document.
-   [Tags in GitHub](https://github.com/apache/airflow/tags) to retrieve the git project sources that
    were used to generate official source packages via git

All those artifacts are not official releases, but they are prepared using officially released sources.
Some of those artifacts are "development" or "pre-release" ones, and they are clearly marked as such
following the ASF Policy.

## User Interface

-   **DAGs**: Overview of all DAGs in your environment.

  ![DAGs](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/dags.png)

-   **Assets**: Overview of Assets with dependencies.

  ![Asset Dependencies](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/assets_graph.png)

-   **Grid**: Grid representation of a DAG that spans across time.

  ![Grid](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/grid.png)

-   **Graph**: Visualization of a DAG's dependencies and their current status for a specific run.

  ![Graph](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/graph.png)

-   **Home**: Summary statistics of your Airflow environment.

  ![Home](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/home.png)

-   **Backfill**: Backfilling a DAG for a specific date range.

  ![Backfill](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/backfill.png)

-   **Code**: Quick way to view source code of a DAG.

  ![Code](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/code.png)

## Semantic Versioning

Airflow follows a strict [SemVer](https://semver.org/) approach.

## Version Life Cycle

Apache Airflow version life cycle:

<!-- This table is automatically updated by prek scripts/ci/prek/supported_versions.py -->
<!-- Beginning of auto-generated table -->

| Version   | Current Patch/Minor   | State     | First Release   | Limited Maintenance   | EOL/Terminated   |
|-----------|-----------------------|-----------|-----------------|-----------------------|------------------|
| 3         | 3.1.0                 | Supported | Apr 22, 2025    | TBD                   | TBD              |
| 2         | 2.11.0                | Supported | Dec 17, 2020    | Oct 22, 2025          | Apr 22, 2026     |
| 1.10      | 1.10.15               | EOL       | Aug 27, 2018    | Dec 17, 2020          | June 17, 2021    |
| 1.9       | 1.9.0                 | EOL       | Jan 03, 2018    | Aug 27, 2018          | Aug 27, 2018     |
| 1.8       | 1.8.2                 | EOL       | Mar 19, 2017    | Jan 03, 2018          | Jan 03, 2018     |
| 1.7       | 1.7.1.2               | EOL       | Mar 28, 2016    | Mar 19, 2017          | Mar 19, 2017     |

<!-- End of auto-generated table -->

## Support for Python and Kubernetes versions

Airflow supports Python and Kubernetes versions based on their official release schedules.

## Base OS support for reference Airflow images

Airflow provides container images with a stable Debian OS base.

## Approach to dependencies of Airflow

Airflow has a detailed approach to managing its dependencies.

## Contributing

Contribute to Apache Airflow.

## Voting Policy

* Commits need a +1 vote from a committer who is not the author
* When we do AIP voting, both PMC member's and committer's `+1s` are considered a binding vote.

## Who uses Apache Airflow?

See the organizations using Apache Airflow.

## Who maintains Apache Airflow?

Learn about the core committers and maintainers of Apache Airflow.

## What goes into the next release?

Learn about which PRs and issues will be released in the next version.

## Can I use the Apache Airflow logo in my presentation?

Yes!  Be sure to abide by the Apache Foundation [trademark policies](https://www.apache.org/foundation/marks/#books) and the Apache Airflow [Brandbook](https://cwiki.apache.org/confluence/display/AIRFLOW/Brandbook). The most up-to-date logos are found in [this repo](https://github.com/apache/airflow/tree/main/airflow-core/docs/img/logos/) and on the Apache Software Foundation [website](https://www.apache.org/logos/about.html).

## Links

*   [Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
*   [Chat](https://s.apache.org/airflow-slack)
*   [Community Information](https://airflow.apache.org/community/)

## Sponsors

The CI infrastructure for Apache Airflow has been sponsored by:

<!-- Ordered by most recently "funded" -->

<a href="https://astronomer.io"><img src="https://assets2.astronomer.io/logos/logoForLIGHTbackground.png" alt="astronomer.io" width="250px"></a>
<a href="https://aws.amazon.com/opensource/"><img src="https://github.com/apache/airflow/blob/main/providers/amazon/docs/integration-logos/AWS-Cloud-alt_light-bg@4x.png?raw=true" alt="AWS OpenSource" width="130px"></a>