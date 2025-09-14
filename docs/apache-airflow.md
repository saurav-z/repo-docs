# Apache Airflow: Orchestrate Workflows with Code

**Apache Airflow is a powerful platform that allows you to programmatically author, schedule, and monitor workflows, making data pipelines reliable and manageable.**  Explore the official repository: [https://github.com/apache/airflow](https://github.com/apache/airflow)

[![License](https://img.shields.io/:license-Apache%202-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0.txt)
[![PyPI version](https://badge.fury.io/py/apache-airflow.svg)](https://badge.fury.io/py/apache-airflow)
[![Docker Pulls](https://img.shields.io/docker/pulls/apache/airflow.svg)](https://hub.docker.com/r/apache/airflow)
[![Contributors](https://img.shields.io/github/contributors/apache/airflow)](https://github.com/apache/airflow/graphs/contributors)
![Commit Activity](https://img.shields.io/github/commit-activity/m/apache/airflow)

<picture width="500">
  <img
    src="https://github.com/apache/airflow/blob/19ebcac2395ef9a6b6ded3a2faa29dc960c1e635/docs/apache-airflow/img/logos/wordmark_1.png?raw=true"
    alt="Apache Airflow logo"
  />
</picture>

**Key Features:**

*   **Dynamic:** Define workflows in code (Python), enabling dynamic DAG generation and parameterization.
*   **Extensible:** Leverage a wide array of built-in operators and customize workflows to fit your needs.
*   **Flexible:** Utilize the Jinja templating engine for rich customizations.
*   **Scalable:** Execute tasks concurrently on an array of workers.
*   **Monitorable:** Visualize pipelines, track progress, and troubleshoot issues through a user-friendly UI.

**Benefits of Using Airflow:**

*   **Maintainability:** Workflows defined as code are easier to understand, modify, and maintain.
*   **Version Control:** Easily track changes and revert to previous versions using your preferred version control system.
*   **Testability:** Implement unit tests and integration tests for your workflows.
*   **Collaboration:** Foster collaboration among data engineers, data scientists, and other stakeholders.

## Table of Contents

- [Project Focus](#project-focus)
- [Principles](#principles)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Installing from PyPI](#installing-from-pypi)
- [Installation](#installation)
- [Official source code](#official-source-code)
- [Convenience packages](#convenience-packages)
- [User Interface](#user-interface)
- [Semantic versioning](#semantic-versioning)
- [Version Life Cycle](#version-life-cycle)
- [Support for Python and Kubernetes versions](#support-for-python-and-kubernetes-versions)
- [Base OS support for reference Airflow images](#base-os-support-for-reference-airflow-images)
- [Approach to dependencies of Airflow](#approach-to-dependencies-of-airflow)
- [Contributing](#contributing)
- [Voting Policy](#voting-policy)
- [Who uses Apache Airflow?](#who-uses-apache-airflow)
- [Who maintains Apache Airflow?](#who-maintains-apache-airflow)
- [What goes into the next release?](#what-goes-into-the-next-release)
- [Can I use the Apache Airflow logo in my presentation?](#can-i-use-the-apache-airflow-logo-in-my-presentation)
- [Links](#links)
- [Sponsors](#sponsors)

## Project Focus

Airflow is best suited for workflows that are mostly static and change slowly, which clarifies the unit of work and continuity. It is designed for data processing, and tasks should ideally be idempotent.

## Principles

Airflow is built on these key principles:

*   **Dynamic:** Code-based pipeline definitions for dynamic DAG generation and parameterization.
*   **Extensible:** Rich framework with built-in operators and extensibility.
*   **Flexible:** Leverage Jinja templating for customization.

## Requirements

Apache Airflow is tested with these versions:

|            | Main version (dev)     | Stable version (3.0.6) |
|------------|------------------------|------------------------|
| Python     | 3.10, 3.11, 3.12, 3.13 | 3.9, 3.10, 3.11, 3.12  |
| Platform   | AMD64/ARM64(\*)        | AMD64/ARM64(\*)        |
| Kubernetes | 1.30, 1.31, 1.32, 1.33 | 1.30, 1.31, 1.32, 1.33 |
| PostgreSQL | 13, 14, 15, 16, 17     | 13, 14, 15, 16, 17     |
| MySQL      | 8.0, 8.4, Innovation   | 8.0, 8.4, Innovation   |
| SQLite     | 3.15.0+                | 3.15.0+                |

\* Experimental

**Notes:**

*   MariaDB is not tested/recommended.
*   SQLite is used in Airflow tests. Do not use it in production.
*   Airflow can run on POSIX-compliant Operating Systems. For development, it's tested on Linux Distros and macOS.
*   Windows is supported via WSL2 (Windows Subsystem for Linux 2) or via Linux Containers but not a high priority.
*   The only distro that is used in the CI tests and the [Community managed DockerHub image](https://hub.docker.com/p/apache/airflow) is `Debian Bookworm`.

## Getting started

Find official documentation for installation, getting started, and tutorials on the [Apache Airflow website](https://airflow.apache.org/docs/apache-airflow/stable/).

> Note: Main branch documentation is on [s.apache.org/airflow-docs](https://s.apache.org/airflow-docs/).

Also explore [Airflow Improvement Proposals (AIPs)](https://cwiki.apache.org/confluence/display/AIRFLOW/Airflow+Improvement+Proposals).

## Installing from PyPI

Install Apache Airflow with `pip` using constraints for repeatable installations.

1.  Install Airflow:

```bash
pip install 'apache-airflow==3.0.6' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.6/constraints-3.10.txt"
```

2.  Install with extras:

```bash
pip install 'apache-airflow[postgres,google]==3.0.6' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.6/constraints-3.10.txt"
```

See [providers](http://airflow.apache.org/docs/apache-airflow-providers/index.html) for provider distribution installation instructions.

## Installation

For detailed setup instructions, see the [INSTALLING.md](INSTALLING.md) file.

## Official source code

Apache Airflow is an Apache Software Foundation (ASF) project, and source code releases:

*   Follow the [ASF Release Policy](https://www.apache.org/legal/release-policy.html)
*   Can be downloaded from [the ASF Distribution Directory](https://downloads.apache.org/airflow)
*   Are cryptographically signed by the release manager
*   Are officially voted on by the PMC members during the [Release Approval Process](https://www.apache.org/legal/release-policy.html#release-approval)

## Convenience packages

Alternative installation methods:

*   [PyPI releases](https://pypi.org/project/apache-airflow/) with `pip`.
*   [Docker Images](https://hub.docker.com/r/apache/airflow) via `docker`.
*   [Tags in GitHub](https://github.com/apache/airflow/tags) for retrieving git project sources.

## User Interface

Airflow offers a rich UI for monitoring and managing workflows:

*   **DAGs**: Overview of all DAGs.
*   **Assets**: Overview of Assets with dependencies.
*   **Grid**: Grid representation of DAGs.
*   **Graph**: DAG dependency visualization.
*   **Home**: Summary statistics.
*   **Backfill**: Backfilling a DAG.
*   **Code**: View DAG source code.

  ![DAGs](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/dags.png)
  ![Asset Dependencies](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/assets_graph.png)
  ![Grid](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/grid.png)
  ![Graph](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/graph.png)
  ![Home](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/home.png)
  ![Backfill](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/backfill.png)
  ![Code](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/code.png)

## Semantic versioning

Airflow 2.0.0 and later use [SemVer](https://semver.org/).

*   **Airflow**: Applies to core Airflow (excluding providers). Dependency version changes aren't breaking changes.
*   **Airflow Providers**: SemVer for provider code changes only. Independent of Airflow version.
*   **Airflow Helm Chart**: SemVer for chart changes. Independent of Airflow version.
*   **Airflow API clients**: Independent from Airflow versions. Follow their own SemVer.

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

*   Limited support includes security and critical bug fixes.
*   EOL versions get no fixes or support.
*   Upgrade to the latest minor release of the major version in use.
*   Upgrade to the latest major release before the EOL date.

## Support for Python and Kubernetes versions

*   EOL versions are dropped.
*   New versions of Python/Kubernetes are supported in `main` after official release and integration in CI.

## Base OS support for reference Airflow images

Airflow provides container images with:

*   Debian OS.
*   Python versions supported at release time.
*   Libraries for database connections.
*   Predefined popular providers.
*   Ability to build custom images.

Debian stable versions are used.  Airflow uses the latest supported OS version, switching before end-of-regular support of previous versions.

## Approach to dependencies of Airflow

Airflow uses `constraints` for repeatable installations, while allowing users to upgrade dependencies.  Dependencies are not upper-bounded by default unless necessary.

Dependencies are:

*   Upper-bound in ``pyproject.toml`` for `SQLAlchemy`, `Alembic`, `Flask`, `werkzeug`, `celery`, and `kubernetes`.
*   `extras` and `providers` dependencies are maintained in `provider.yaml`, not upper-bound by default.

## Contributing

Contribute to Apache Airflow with the [contributors' guide](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst).  Check out the [contribution quickstart](https://github.com/apache/airflow/blob/main/contributing-docs/03_contributors_quick_start.rst).

## Voting Policy

*   Commits need a +1 vote from a committer.
*   AIP voting requires both PMC and committer `+1s`.

## Who uses Apache Airflow?

Around 500 organizations use Apache Airflow, see [INTHEWILD.md](https://github.com/apache/airflow/blob/main/INTHEWILD.md).  Add your organization with a PR.

## Who maintains Apache Airflow?

Airflow is community-driven.  The [core committers/maintainers](https://people.apache.org/committers-by-project.html#airflow) review and merge PRs.  See the [committer requirements](https://github.com/apache/airflow/blob/main/COMMITTERS.rst#guidelines-to-become-an-airflow-committer).

## What goes into the next release?

The release process and milestone assignments are explained in the [dev/WHAT_GOES_INTO_THE_NEXT_RELEASE.md](dev/WHAT_GOES_INTO_THE_NEXT_RELEASE.md) document.

## Can I use the Apache Airflow logo in my presentation?

Yes, follow the Apache Foundation [trademark policies](https://www.apache.org/foundation/marks/#books) and the Apache Airflow [Brandbook](https://cwiki.apache.org/confluence/display/AIRFLOW/Brandbook). Logos are found in [this repo](https://github.com/apache/airflow/tree/main/airflow-core/docs/img/logos/).

## Links

*   [Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
*   [Chat](https://s.apache.org/airflow-slack)
*   [Community Information](https://airflow.apache.org/community/)

## Sponsors

Sponsored by:

<a href="https://astronomer.io"><img src="https://assets2.astronomer.io/logos/logoForLIGHTbackground.png" alt="astronomer.io" width="250px"></a>
<a href="https://aws.amazon.com/opensource/"><img src="https://github.com/apache/airflow/blob/main/providers/amazon/docs/integration-logos/AWS-Cloud-alt_light-bg@4x.png?raw=true" alt="AWS OpenSource" width="130px"></a>