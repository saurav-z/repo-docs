# Apache Airflow: Orchestrate Your Workflows with Code

**Apache Airflow empowers you to programmatically author, schedule, and monitor complex workflows.**

[![License](https://img.shields.io/:license-Apache%202-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0.txt)
[![PyPI version](https://badge.fury.io/py/apache-airflow.svg)](https://badge.fury.io/py/apache-airflow)
[![Docker Pulls](https://img.shields.io/docker/pulls/apache/airflow.svg)](https://hub.docker.com/r/apache/airflow)
[![Contributors](https://img.shields.io/github/contributors/apache/airflow)](https://github.com/apache/airflow/graphs/contributors)
[![Slack Status](https://img.shields.io/badge/slack-join_chat-white.svg?logo=slack&style=social)](https://s.apache.org/airflow-slack)

[View the original repository on GitHub](https://github.com/apache/airflow)

Apache Airflow is a powerful, open-source platform for orchestrating and scheduling complex data pipelines.  It allows you to define your workflows as code, making them more manageable, versionable, and collaborative.

**Key Features:**

*   **Dynamic Pipelines:** Define workflows as code using Python, enabling dynamic DAG generation and parameterization.
*   **Extensible Architecture:** Leverage a wide range of built-in operators and easily extend Airflow to fit your specific needs.
*   **Flexible Templating:** Utilize the Jinja templating engine for rich customizations and advanced pipeline control.
*   **Comprehensive Monitoring:**  Monitor pipeline execution, visualize dependencies, and troubleshoot issues through a rich user interface.
*   **Scalability:** Designed to handle large-scale workflows.

<picture width="500">
  <img
    src="https://github.com/apache/airflow/blob/19ebcac2395ef9a6b6ded3a2faa29dc960c1e635/docs/apache-airflow/img/logos/wordmark_1.png?raw=true"
    alt="Apache Airflow logo"
  />
</picture>

## Table of Contents

*   [Project Focus](#project-focus)
*   [Principles](#principles)
*   [Requirements](#requirements)
*   [Getting Started](#getting-started)
*   [Installing from PyPI](#installing-from-pypi)
*   [Installation](#installation)
*   [Official Source Code](#official-source-code)
*   [Convenience Packages](#convenience-packages)
*   [User Interface](#user-interface)
*   [Semantic Versioning](#semantic-versioning)
*   [Version Life Cycle](#version-life-cycle)
*   [Support for Python and Kubernetes Versions](#support-for-python-and-kubernetes-versions)
*   [Base OS Support for Reference Airflow Images](#base-os-support-for-reference-airflow-images)
*   [Approach to Dependencies of Airflow](#approach-to-dependencies-of-airflow)
*   [Contributing](#contributing)
*   [Voting Policy](#voting-policy)
*   [Who Uses Apache Airflow?](#who-uses-apache-airflow)
*   [Who Maintains Apache Airflow?](#who-maintains-apache-airflow)
*   [What Goes into the Next Release?](#what-goes-into-the-next-release)
*   [Can I use the Apache Airflow logo in my presentation?](#can-i-use-the-apache-airflow-logo-in-my-presentation)
*   [Links](#links)
*   [Sponsors](#sponsors)

## Project Focus

Airflow excels with workflows that are relatively static and evolve gradually.  It's ideal for defining units of work and maintaining consistency across runs.

Airflow is commonly used to process data, with a focus on idempotent tasks. For high-volume data-intensive tasks, it's designed to integrate with specialized external services.

Airflow is not a streaming solution, but it is often used to process real-time data, pulling data off streams in batches.

## Principles

*   **Dynamic:** Pipelines are defined in code, enabling dynamic dag generation and parameterization.
*   **Extensible:** The Airflow framework includes a wide range of built-in operators and can be extended to fit your needs.
*   **Flexible:** Airflow leverages the [**Jinja**](https://jinja.palletsprojects.com) templating engine, allowing rich customizations.

## Requirements

Apache Airflow is tested with:

|            | Main version (dev)     | Stable version (3.0.6) |
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

For installation and usage guides, visit the official Airflow website documentation: [Stable Documentation](https://airflow.apache.org/docs/apache-airflow/stable/).  Access [the tutorial](https://airflow.apache.org/docs/apache-airflow/stable/tutorial/) to learn more.

> Note: If you're looking for documentation for the main branch (latest development branch): you can find it on [s.apache.org/airflow-docs](https://s.apache.org/airflow-docs/).

For more information on Airflow Improvement Proposals (AIPs), visit
the [Airflow Wiki](https://cwiki.apache.org/confluence/display/AIRFLOW/Airflow+Improvement+Proposals).

Documentation for dependent projects like provider distributions, Docker image, Helm Chart, you'll find it in [the documentation index](https://airflow.apache.org/docs/).

## Installing from PyPI

Install Apache Airflow from PyPI using the `apache-airflow` package; follow these guidelines for reliable installations:

1.  **Install just Airflow:**

    ```bash
    pip install 'apache-airflow==3.0.6' \
     --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.6/constraints-3.10.txt"
    ```

2.  **Install with Extras (e.g., postgres, google):**

    ```bash
    pip install 'apache-airflow[postgres,google]==3.0.6' \
     --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.6/constraints-3.10.txt"
    ```

For more details on installing provider distributions, please refer to the [providers](http://airflow.apache.org/docs/apache-airflow-providers/index.html) documentation.

## Installation

Detailed setup and installation instructions are available in the [INSTALLING.md](INSTALLING.md) file.

## Official Source Code

Apache Airflow is an [Apache Software Foundation](https://www.apache.org) (ASF) project,
and our official source code releases:

-   Follow the [ASF Release Policy](https://www.apache.org/legal/release-policy.html)
-   Can be downloaded from [the ASF Distribution Directory](https://downloads.apache.org/airflow)
-   Are cryptographically signed by the release manager
-   Are officially voted on by the PMC members during the
    [Release Approval Process](https://www.apache.org/legal/release-policy.html#release-approval)

Following the ASF rules, the source packages released must be sufficient for a user to build and test the
release provided they have access to the appropriate platform and tools.

## Convenience Packages

Explore alternative Airflow installation methods beyond official releases:

-   [PyPI releases](https://pypi.org/project/apache-airflow/) for standard `pip` installation.
-   [Docker Images](https://hub.docker.com/r/apache/airflow) using `docker` for Kubernetes, Helm Charts, etc. Learn more in the [Latest docs](https://airflow.apache.org/docs/docker-stack/index.html) and [images](https://airflow.apache.org/docs/docker-stack/index.html) document.
-   [Tags in GitHub](https://github.com/apache/airflow/tags) for retrieving project sources via git.

These artifacts are not official releases, but they are prepared using officially released sources. Some are marked as "development" or "pre-release."

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

Airflow 2.0.0+ strictly adheres to [SemVer](https://semver.org/) for all released packages.

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

Airflow 2.0+ follows these rules for Python and Kubernetes support:

1.  Drop support for EOL Python/Kubernetes versions.
2.  Support new Python/Kubernetes versions in `main` after official release and CI setup.
3.  This is a best-effort policy, with potential for earlier termination if needed.

## Base OS Support for Reference Airflow Images

The Airflow Community provides container images for each release. Those images contain:

*   Base OS (stable Debian)
*   Base Python installations
*   Libraries for Database connections
*   Predefined popular providers
*   Custom image building support

The base OS version uses stable Debian. Airflow uses all active stable versions.
For example switch from ``Debian Bullseye`` to ``Debian Bookworm`` has been implemented
before 2.8.0 release in October 2023 and ``Debian Bookworm`` will be the only option supported as of
Airflow 2.10.0.

## Approach to Dependencies of Airflow

Airflow's approach to dependencies balances application stability with the need for flexible DAG development.

-   `constraints` are used for repeatable installations.
-   Dependencies are not upper-bound by default unless there's a strong reason.

### Approach for dependencies for Airflow Core

These dependencies are maintained in `pyproject.toml`.

Key dependencies are upper-bound for stability:

*   `SQLAlchemy`: Upper-bound to specific MINOR version.
*   `Alembic`: Stable in MINOR versions
*   `Flask`: Limited to MAJOR version.
*   `werkzeug`:
*   `celery`: Celery is a crucial component of Airflow as it used for CeleryExecutor (and similar). Celery [follows SemVer](https://docs.celeryq.dev/en/stable/contributing.html?highlight=semver#versions), so
   we should upper-bound it to the next MAJOR version. Also, when we bump the upper version of the library,
   we should make sure Celery Provider minimum Airflow version is updated.
*   `kubernetes`: Kubernetes is a crucial component of Airflow as it is used for the KubernetesExecutor
   (and similar). Kubernetes Python library [follows SemVer](https://github.com/kubernetes-client/python#compatibility),
   so we should upper-bound it to the next MAJOR version. Also, when we bump the upper version of the library,
   we should make sure Kubernetes Provider minimum Airflow version is updated.

### Approach for dependencies in Airflow Providers and extras

Those dependencies are maintained in `provider.yaml` of each provider.

By default, we should not upper-bound dependencies for providers, however each provider's maintainer
might decide to add additional limits (and justify them with comment).

## Contributing

Contribute to Apache Airflow!  See the [contributors' guide](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst) and [contribution quickstart](https://github.com/apache/airflow/blob/main/contributing-docs/03_contributors_quick_start.rst) for details.

Official Docker (container) images for Apache Airflow are described in [images](https://github.com/apache/airflow/blob/main/dev/breeze/doc/ci/02_images.md).

## Voting Policy

* Commits need a +1 vote from a committer who is not the author
* When we do AIP voting, both PMC member's and committer's `+1s` are considered a binding vote.

## Who Uses Apache Airflow?

Many organizations, including approximately 500 known companies, use Apache Airflow.  See the [INTHEWILD.md](https://github.com/apache/airflow/blob/main/INTHEWILD.md) file for a list.

## Who Maintains Apache Airflow?

The Airflow community drives development, with [core committers/maintainers](https://people.apache.org/committers-by-project.html#airflow) responsible for code review and merging.  See the [committer requirements](https://github.com/apache/airflow/blob/main/COMMITTERS.rst#guidelines-to-become-an-airflow-committer) for more information on becoming a maintainer.

## What Goes into the Next Release?

The timing of PRs and issues in relation to the next release cycle is explained, see the detailed guide [What goes into the next release](dev/WHAT_GOES_INTO_THE_NEXT_RELEASE.md).

## Can I use the Apache Airflow logo in my presentation?

Yes, but please adhere to the Apache Foundation [trademark policies](https://www.apache.org/foundation/marks/#books) and the Apache Airflow [Brandbook](https://cwiki.apache.org/confluence/display/AIRFLOW/Brandbook).  Find the latest logos in [this repo](https://github.com/apache/airflow/tree/main/airflow-core/docs/img/logos/) and on the Apache Software Foundation [website](https://www.apache.org/logos/about.html).

## Links

*   [Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
*   [Chat](https://s.apache.org/airflow-slack)
*   [Community Information](https://airflow.apache.org/community/)

## Sponsors

The CI infrastructure for Apache Airflow has been sponsored by:

<!-- Ordered by most recently "funded" -->

<a href="https://astronomer.io"><img src="https://assets2.astronomer.io/logos/logoForLIGHTbackground.png" alt="astronomer.io" width="250px"></a>
<a href="https://aws.amazon.com/opensource/"><img src="https://github.com/apache/airflow/blob/main/providers/amazon/docs/integration-logos/AWS-Cloud-alt_light-bg@4x.png?raw=true" alt="AWS OpenSource" width="130px"></a>