# Apache Airflow: Orchestrate and Schedule Your Workflows with Code

**Automate, monitor, and scale your data pipelines with Apache Airflow, the open-source workflow management platform.**

[![License](https://img.shields.io/:license-Apache%202-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0.txt)
[![PyPI version](https://badge.fury.io/py/apache-airflow.svg)](https://badge.fury.io/py/apache-airflow)
[![Docker Pulls](https://img.shields.io/docker/pulls/apache/airflow.svg)](https://hub.docker.com/r/apache/airflow)
[![Contributors](https://img.shields.io/github/contributors/apache/airflow)](https://github.com/apache/airflow/graphs/contributors)

[Get started with Apache Airflow](https://github.com/apache/airflow) and experience the power of programmatic workflow management.

## Key Features

*   **Dynamic Workflows:** Define pipelines as code for flexibility and control.
*   **Extensible Architecture:** Leverage a wide range of built-in operators and easily create custom integrations.
*   **Scalable Execution:** Airflow efficiently schedules and executes tasks across a distributed worker environment.
*   **Rich UI:** Monitor, visualize, and troubleshoot your workflows with an intuitive web interface.
*   **Version Control & Collaboration:** Benefit from code-based workflows, enabling versioning, testing, and collaboration.

## Core Principles

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

Explore the official Apache Airflow documentation for comprehensive guidance:

*   [Installation](https://airflow.apache.org/docs/apache-airflow/stable/installation/)
*   [Getting Started](https://airflow.apache.org/docs/apache-airflow/stable/start.html)
*   [Tutorial](https://airflow.apache.org/docs/apache-airflow/stable/tutorial/)

For further resources, including Airflow Improvement Proposals (AIPs) and documentation for dependent projects, visit the [Airflow documentation index](https://airflow.apache.org/docs/).

## Installing from PyPI

To install Apache Airflow using pip, run the following:

```bash
pip install 'apache-airflow==3.0.6' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.6/constraints-3.10.txt"
```

**Install with extras (example postgres, google):**

```bash
pip install 'apache-airflow[postgres,google]==3.0.6' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.6/constraints-3.10.txt"
```

For details, consult the [installing from PyPI](https://github.com/apache/airflow#installing-from-pypi) section of the README.

## Installation

For detailed setup instructions, refer to the [INSTALLING.md](INSTALLING.md) file.

## Official Source Code

Apache Airflow is an Apache Software Foundation (ASF) project, with official source code releases:

*   Released under the [ASF Release Policy](https://www.apache.org/legal/release-policy.html)
*   Available for download from [the ASF Distribution Directory](https://downloads.apache.org/airflow)
*   Cryptographically signed by release managers
*   Subject to official PMC member voting during the [Release Approval Process](https://www.apache.org/legal/release-policy.html#release-approval)

## Convenience Packages

Apache Airflow offers various installation options beyond official releases:

*   [PyPI Releases](https://pypi.org/project/apache-airflow/) - Standard `pip` installation.
*   [Docker Images](https://hub.docker.com/r/apache/airflow) - Install via `docker`. Read about using, customizing, and extending the images in the [Latest docs](https://airflow.apache.org/docs/docker-stack/index.html).
*   [GitHub Tags](https://github.com/apache/airflow/tags) - Git project sources to generate official source packages.

## User Interface

*   **DAGs:** Overview of all DAGs in your environment.

    ![DAGs](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/dags.png)

*   **Assets**: Overview of Assets with dependencies.

    ![Asset Dependencies](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/assets_graph.png)

*   **Grid:** Grid representation of a DAG that spans across time.

    ![Grid](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/grid.png)

*   **Graph:** Visualization of a DAG's dependencies and their current status for a specific run.

    ![Graph](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/graph.png)

*   **Home:** Summary statistics of your Airflow environment.

    ![Home](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/home.png)

*   **Backfill:** Backfilling a DAG for a specific date range.

    ![Backfill](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/backfill.png)

*   **Code:** Quick way to view source code of a DAG.

    ![Code](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/code.png)

## Semantic Versioning

As of Airflow 2.0.0, Airflow follows [SemVer](https://semver.org/) for all packages released.
See the [Semantic Versioning](https://github.com/apache/airflow#semantic-versioning) for details.

## Version Life Cycle

Check the [Version Life Cycle](https://github.com/apache/airflow#version-life-cycle) for details on supported and EOL versions.

## Support for Python and Kubernetes versions

See the [Support for Python and Kubernetes versions](https://github.com/apache/airflow#support-for-python-and-kubernetes-versions) for details.

## Base OS Support for reference Airflow images

See the [Base OS support for reference Airflow images](https://github.com/apache/airflow#base-os-support-for-reference-airflow-images) for details.

## Approach to dependencies of Airflow

See the [Approach to dependencies of Airflow](https://github.com/apache/airflow#approach-to-dependencies-of-airflow) for details.

## Contributing

Join the Apache Airflow community!  See the [contributors' guide](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst) and [contribution quickstart](https://github.com/apache/airflow/blob/main/contributing-docs/03_contributors_quick_start.rst) for how to get started.

## Voting Policy

*   Commits require a +1 vote from a committer who is not the author
*   AIP voting requires a binding vote from both PMC members and committers.

## Who uses Apache Airflow?

Discover a list of organizations [using Apache Airflow](https://github.com/apache/airflow/blob/main/INTHEWILD.md).

## Who maintains Apache Airflow?

Apache Airflow is maintained by a [community](https://github.com/apache/airflow/graphs/contributors), under the guidance of [core committers/maintainers](https://people.apache.org/committers-by-project.html#airflow).  Learn more about becoming a committer in the [committer requirements](https://github.com/apache/airflow/blob/main/COMMITTERS.rst#guidelines-to-become-an-airflow-committer).

## What goes into the next release?

For insights, see the [WHAT_GOES_INTO_THE_NEXT_RELEASE.md](dev/WHAT_GOES_INTO_THE_NEXT_RELEASE.md) document.

## Can I use the Apache Airflow logo in my presentation?

Yes, but please adhere to Apache Foundation [trademark policies](https://www.apache.org/foundation/marks/#books) and the [Apache Airflow Brandbook](https://cwiki.apache.org/confluence/display/AIRFLOW/Brandbook).

## Links

*   [Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
*   [Chat](https://s.apache.org/airflow-slack)
*   [Community Information](https://airflow.apache.org/community/)

## Sponsors

[Sponsors](https://github.com/apache/airflow#sponsors)