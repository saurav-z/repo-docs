# Apache Airflow: Orchestrate Your Workflows with Code

**Apache Airflow is a powerful platform for programmatically authoring, scheduling, and monitoring data pipelines, transforming complex workflows into maintainable and scalable solutions.** [Explore the Apache Airflow Repository](https://github.com/apache/airflow)

[![License](https://img.shields.io/:license-Apache%202-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0.txt)
[![PyPI version](https://badge.fury.io/py/apache-airflow.svg)](https://badge.fury.io/py/apache-airflow)
[![Docker Pulls](https://img.shields.io/docker/pulls/apache/airflow.svg)](https://hub.docker.com/r/apache/airflow)
[![Contributors](https://img.shields.io/github/contributors/apache/airflow)](https://github.com/apache/airflow/graphs/contributors)
[![Slack Status](https://img.shields.io/badge/slack-join_chat-white.svg?logo=slack&style=social)](https://s.apache.org/airflow-slack)

**Key Features:**

*   **Workflow-as-Code:** Define pipelines using Python, enabling version control, testing, and collaboration.
*   **Dynamic Pipelines:**  Generate and customize pipelines dynamically using Jinja templating.
*   **Extensible Architecture:**  Utilize a vast library of built-in operators and easily extend Airflow to meet specific needs.
*   **Scalable Scheduling:**  Orchestrate tasks efficiently across distributed workers with a robust scheduler.
*   **Rich UI:** Monitor pipeline progress, troubleshoot issues, and visualize workflows through a user-friendly web interface.

## Core Concepts

Apache Airflow excels with workflows that are primarily static and evolve gradually, enhancing the clarity and continuity of the unit of work.  While Airflow is often used for data processing, it emphasizes idempotent tasks, which avoid data duplication, and can delegate high-volume tasks to specialized services.

## Key Principles

*   **Dynamic:** Define your pipelines in code, which supports dynamic DAG generation and parameterization.
*   **Extensible:**  Airflow's framework is built with a broad array of built-in operators and it is extendable to fit any need.
*   **Flexible:** Airflow leverages the [**Jinja**](https://jinja.palletsprojects.com) templating engine, allowing rich customizations.

## Requirements

Apache Airflow is tested with the following versions of Python, Kubernetes, and databases.

| Feature      | Main Version (dev) | Stable Version (3.0.5) |
|--------------|--------------------|------------------------|
| Python       | 3.10, 3.11, 3.12, 3.13 | 3.9, 3.10, 3.11, 3.12  |
| Platform     | AMD64/ARM64(\*)    | AMD64/ARM64(\*)        |
| Kubernetes   | 1.30, 1.31, 1.32, 1.33 | 1.30, 1.31, 1.32, 1.33 |
| PostgreSQL   | 13, 14, 15, 16, 17   | 13, 14, 15, 16, 17     |
| MySQL        | 8.0, 8.4, Innovation | 8.0, 8.4, Innovation   |
| SQLite       | 3.15.0+            | 3.15.0+                |

\* Experimental

**Note:** MariaDB is not tested/recommended.

**Note:** SQLite is used in Airflow tests. Do not use it in production. We recommend
using the latest stable version of SQLite for local development.

**Note:** Airflow currently can be run on POSIX-compliant Operating Systems. For development, it is regularly
tested on fairly modern Linux Distros and recent versions of macOS.
On Windows you can run it via WSL2 (Windows Subsystem for Linux 2) or via Linux Containers.
The work to add Windows support is tracked via [#10388](https://github.com/apache/airflow/issues/10388), but
it is not a high priority. You should only use Linux-based distros as "Production" execution environment
as this is the only environment that is supported. The only distro that is used in our CI tests and that
is used in the [Community managed DockerHub image](https://hub.docker.com/p/apache/airflow) is
`Debian Bookworm`.

## Getting Started

Explore the official Apache Airflow website ([latest stable release](https://airflow.apache.org/docs/apache-airflow/stable/)) for complete documentation, installation instructions, tutorials, and more.

## Installation

For detailed instructions, refer to the [INSTALLING.md](INSTALLING.md) file.

## Installing from PyPI

Install Apache Airflow using `pip` and carefully review the constraints files to ensure a repeatable and compatible installation.

1.  Installing just Airflow:

```bash
pip install 'apache-airflow==3.0.5' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.5/constraints-3.10.txt"
```

2.  Installing with extras (i.e., postgres, google):

```bash
pip install 'apache-airflow[postgres,google]==3.0.5' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.5/constraints-3.10.txt"
```

For information on installing provider distributions, check
[providers](http://airflow.apache.org/docs/apache-airflow-providers/index.html).

## Official Source Code

Apache Airflow is an Apache Software Foundation (ASF) project with source code releases that:

*   Adhere to the [ASF Release Policy](https://www.apache.org/legal/release-policy.html)
*   Are available for download from [the ASF Distribution Directory](https://downloads.apache.org/airflow)
*   Are cryptographically signed by the release manager
*   Are officially voted on by the PMC members during the [Release Approval Process](https://www.apache.org/legal/release-policy.html#release-approval)

## Convenience Packages

The project offers a wide range of convenient installation and deployment methods:

*   **PyPI Releases:** Use `pip` to install Airflow.
*   **Docker Images:** Deploy Airflow using Docker, Kubernetes, or Helm Charts. Details on the images are in the [images](https://airflow.apache.org/docs/docker-stack/index.html) document.
*   **GitHub Tags:** Access project sources via tags in GitHub.

## User Interface

The Airflow UI provides a comprehensive view of your workflows.

*   **DAGs:** Overview of all DAGs in your environment.

  ![DAGs](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/dags.png)

*   **Assets:** Overview of Assets with dependencies.

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

Airflow utilizes [SemVer](https://semver.org/) for package releases, providing predictable versioning for core functionality.

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

Limited support versions will be supported with security and critical bug fix only.
EOL versions will not get any fixes nor support.
We always recommend that all users run the latest available minor release for whatever major version is in use.
We **highly** recommend upgrading to the latest Airflow major release at the earliest convenient time and before the EOL date.

## Contributing

Join the Apache Airflow community! Review the [contributors' guide](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst) for instructions, and explore the [contribution quickstart](https://github.com/apache/airflow/blob/main/contributing-docs/03_contributors_quick_start.rst).

## Who Uses Apache Airflow?

Many organizations use Airflow, with nearly 500 publicly known ones listed [here](https://github.com/apache/airflow/blob/main/INTHEWILD.md).

## Who Maintains Apache Airflow?

Airflow is a community-driven project maintained by [core committers/maintainers](https://people.apache.org/committers-by-project.html#airflow). Review the Apache Airflow [committer requirements](https://github.com/apache/airflow/blob/main/COMMITTERS.rst#guidelines-to-become-an-airflow-committer).

## What goes into the next release?

For more information, check out the
[What goes into the next release](dev/WHAT_GOES_INTO_THE_NEXT_RELEASE.md) document in the `dev` folder of the
repository.

## Can I use the Apache Airflow logo in my presentation?

Yes, as long as you adhere to Apache Foundation [trademark policies](https://www.apache.org/foundation/marks/#books) and the Apache Airflow [Brandbook](https://cwiki.apache.org/confluence/display/AIRFLOW/Brandbook).

## Links

*   [Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
*   [Chat](https://s.apache.org/airflow-slack)
*   [Community Information](https://airflow.apache.org/community/)

## Sponsors

<a href="https://astronomer.io"><img src="https://assets2.astronomer.io/logos/logoForLIGHTbackground.png" alt="astronomer.io" width="250px"></a>
<a href="https://aws.amazon.com/opensource/"><img src="https://github.com/apache/airflow/blob/main/providers/amazon/docs/integration-logos/AWS-Cloud-alt_light-bg@4x.png?raw=true" alt="AWS OpenSource" width="130px"></a>