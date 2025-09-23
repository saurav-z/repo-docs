# Apache Airflow: Orchestrate Your Workflows with Code 🚀

**Apache Airflow** is a powerful, open-source platform that allows you to programmatically author, schedule, and monitor your workflows. [Explore the official Apache Airflow repository](https://github.com/apache/airflow).

## Key Features

*   **Dynamic Workflow Definition:** Define workflows as code (Python), enabling dynamic DAG generation and parameterization.
*   **Extensible Architecture:**  Airflow's framework boasts a wide range of built-in operators and flexible extensibility to fit your needs.
*   **Flexible Templating:** Leverage the Jinja templating engine for rich customizations, empowering you with unparalleled control.
*   **Scalable Scheduling & Execution:**  The Airflow scheduler efficiently executes tasks across a distributed worker pool.
*   **Comprehensive Monitoring & UI:**  Visualize pipelines, monitor progress, and troubleshoot with a user-friendly interface.

## Project Focus

Airflow excels at managing workflows that are largely static and evolve gradually. Define DAGs (Directed Acyclic Graphs) in code, ensuring maintainability, version control, testability, and collaboration.

## Principles

*   **Dynamic:** Pipelines are defined in code, enabling dynamic dag generation and parameterization.
*   **Extensible:** The Airflow framework includes a wide range of built-in operators and can be extended to fit your needs.
*   **Flexible:** Airflow leverages the [**Jinja**](https://jinja.palletsprojects.com) templating engine, allowing rich customizations.

## Requirements

Apache Airflow is tested with the following versions:

|            | Main version (dev)           | Stable version (3.0.6) |
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

Visit the official Airflow website documentation (latest **stable** release) for help with
[installing Airflow](https://airflow.apache.org/docs/apache-airflow/stable/installation/),
[getting started](https://airflow.apache.org/docs/apache-airflow/stable/start.html), or walking
through a more complete [tutorial](https://airflow.apache.org/docs/apache-airflow/stable/tutorial/).

> Note: If you're looking for documentation for the main branch (latest development branch): you can find it on [s.apache.org/airflow-docs](https://s.apache.org/airflow-docs/).

For more information on Airflow Improvement Proposals (AIPs), visit
the [Airflow Wiki](https://cwiki.apache.org/confluence/display/AIRFLOW/Airflow+Improvement+Proposals).

Documentation for dependent projects like provider distributions, Docker image, Helm Chart, you'll find it in [the documentation index](https://airflow.apache.org/docs/).

## Installing from PyPI

Install Apache Airflow using the following command:

```bash
pip install 'apache-airflow==3.0.6' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.6/constraints-3.10.txt"
```

To install with extras (e.g., postgres, google), use:

```bash
pip install 'apache-airflow[postgres,google]==3.0.6' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.6/constraints-3.10.txt"
```

For information on installing provider distributions, check
[providers](http://airflow.apache.org/docs/apache-airflow-providers/index.html).

## Installation

For comprehensive instructions on setting up your local development environment and installing Apache Airflow, please refer to the [INSTALLING.md](INSTALLING.md) file.

## Official Source Code

Apache Airflow is an [Apache Software Foundation](https://www.apache.org) (ASF) project, and official source code releases follow the ASF Release Policy and are cryptographically signed.

## Convenience Packages

Airflow offers multiple installation methods for ease of use, including PyPI releases, Docker images, and Git tags.

## User Interface

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

## Semantic Versioning

Airflow follows a strict SemVer approach for all releases starting with version 2.0.0.

## Version Life Cycle

Apache Airflow version life cycle:

<!-- This table is automatically updated by prek scripts/ci/prek/supported_versions.py -->
<!-- Beginning of auto-generated table -->

| Version   | Current Patch/Minor   | State     | First Release   | Limited Maintenance   | EOL/Terminated   |
|-----------|-----------------------|-----------|-----------------|-----------------------|------------------|
| 3         | 3.0.6                 | Supported | Apr 22, 2025    | TBD                   | TBD              |
| 2         | 2.11.0                | Supported | Dec 17, 2020    | Oct 22, 2025          | Apr 22, 2026     |
| 1.10      | 1.10.15               | EOL       | Aug 27, 2018    | Dec 17, 2020          | June 17, 2021    |
| 1.9       | 1.9.0                 | EOL       | Jan 03, 2018    | Aug 27, 2018          | Aug 27, 2018     |
| 1.8       | 1.8.2                 | EOL       | Mar 19, 2017    | Jan 03, 2018          | Jan 03, 2018     |
| 1.7       | 1.7.1.2               | EOL       | Mar 28, 2016    | Mar 19, 2017          | Mar 19, 2017     |

<!-- End of auto-generated table -->

Limited support versions will be supported with security and critical bug fix only.
EOL versions will not get any fixes nor support.
We always recommend that all users run the latest available minor release for whatever major version is in use.
We **highly** recommend upgrading to the latest Airflow major release at the earliest convenient time and before the EOL date.

## Support for Python and Kubernetes Versions

Airflow adheres to a policy based on Python and Kubernetes release schedules, ensuring compatibility and timely updates.

## Base OS Support for Reference Airflow Images

The Airflow Community provides conveniently packaged container images that are published whenever
we publish an Apache Airflow release. Those images contain:

* Base OS with necessary packages to install Airflow (stable Debian OS)
* Base Python installation in versions supported at the time of release for the MINOR version of
  Airflow released (so there could be different versions for 2.3 and 2.2 line for example)
* Libraries required to connect to supported Databases (again the set of databases supported depends
  on the MINOR version of Airflow)
* Predefined set of popular providers (for details see the [Dockerfile](https://raw.githubusercontent.com/apache/airflow/main/Dockerfile)).
* Possibility of building your own, custom image where the user can choose their own set of providers
  and libraries (see [Building the image](https://airflow.apache.org/docs/docker-stack/build.html))
* In the future Airflow might also support a "slim" version without providers nor database clients installed

The version of the base OS image is the stable version of Debian. Airflow supports using all currently active
stable versions - as soon as all Airflow dependencies support building, and we set up the CI pipeline for
building and testing the OS version. Approximately 6 months before the end-of-regular support of a
previous stable version of the OS, Airflow switches the images released to use the latest supported
version of the OS.

For example switch from ``Debian Bullseye`` to ``Debian Bookworm`` has been implemented
before 2.8.0 release in October 2023 and ``Debian Bookworm`` will be the only option supported as of
Airflow 2.10.0.

Users will continue to be able to build their images using stable Debian releases until the end of regular
support and building and verifying of the images happens in our CI but no unit tests were executed using
this image in the `main` branch.

## Approach to Dependencies of Airflow

Airflow uses a constraints-based approach to dependency management, balancing stability and flexibility for users.

### Approach for Dependencies for Airflow Core

Airflow core dependencies are managed in `pyproject.toml`, and several dependencies are upper-bound for stability.

### Approach for Dependencies in Airflow Providers and Extras

Provider dependencies are maintained in each provider's `provider.yaml` file.

## Contributing

Contribute to Apache Airflow by reviewing our [contributors' guide](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst).

## Voting Policy

*   Commits need a +1 vote from a committer who is not the author
*   When we do AIP voting, both PMC member's and committer's `+1s` are considered a binding vote.

## Who Uses Apache Airflow?

Apache Airflow is trusted by a wide range of organizations, and you can find a list [in the wild](https://github.com/apache/airflow/blob/main/INTHEWILD.md).

## Who Maintains Apache Airflow?

The Airflow community drives the project, with [core committers/maintainers](https://people.apache.org/committers-by-project.html#airflow) leading code reviews, feature requests, and more.

## What goes into the next release?

Learn more in the [dev/WHAT_GOES_INTO_THE_NEXT_RELEASE.md](dev/WHAT_GOES_INTO_THE_NEXT_RELEASE.md) document.

## Can I use the Apache Airflow logo in my presentation?

Yes, follow the Apache Foundation trademark policies and the Apache Airflow [Brandbook](https://cwiki.apache.org/confluence/display/AIRFLOW/Brandbook).

## Links

*   [Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
*   [Chat](https://s.apache.org/airflow-slack)
*   [Community Information](https://airflow.apache.org/community/)

## Sponsors

*   [Astronome](https://astronomer.io)
*   [AWS OpenSource](https://aws.amazon.com/opensource/)