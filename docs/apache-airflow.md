# Apache Airflow: Orchestrate Your Workflows with Code

**Automate, schedule, and monitor your data pipelines with Apache Airflow, the leading platform for programmatic workflow management.**  [Visit the original repo](https://github.com/apache/airflow)

<picture width="500">
  <img
    src="https://github.com/apache/airflow/blob/19ebcac2395ef9a6b6ded3a2faa29dc960c1e635/docs/apache-airflow/img/logos/wordmark_1.png?raw=true"
    alt="Apache Airflow logo"
  />
</picture>

Apache Airflow (Airflow) is a powerful platform for defining, scheduling, and monitoring complex workflows.  Built on the principle of "workflows as code," Airflow enables you to build data pipelines that are versionable, testable, and collaborative.

**Key Features:**

*   **Dynamic Workflows:** Define pipelines using Python code for dynamic generation and parameterization.
*   **Extensibility:**  A wide array of built-in operators and easily extended to meet your needs.
*   **Flexible:**  Leverages the Jinja templating engine for rich customizations.
*   **Monitoring & Management**: Rich user interface to visualize, monitor and troubleshoot pipelines.
*   **Idempotency**: Airflow tasks are ideally idempotent.

## Core Concepts

*   **DAGs (Directed Acyclic Graphs):** Workflows are defined as code, enabling dynamic DAG generation and parameterization.
*   **Tasks**: Units of work within a DAG.
*   **Operators**: Pre-built components for common tasks (e.g., executing SQL queries, running Python scripts).
*   **Scheduler**: Executes tasks based on defined dependencies and schedules.
*   **Web UI**: Provides a user-friendly interface for monitoring, managing, and troubleshooting workflows.

## Key Advantages

*   **Version Control**: Store DAGs in a version control system (e.g., Git) for easy tracking of changes and rollbacks.
*   **Scalability**: Distribute task execution across a cluster of workers.
*   **Idempotency**: Design tasks to be idempotent.

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

Get started with Airflow:

*   **Installation**:  Detailed setup instructions in the [INSTALLING.md](INSTALLING.md) file.
*   **Official Documentation**:  Explore installation, tutorials, and comprehensive information on the [Airflow Documentation](https://airflow.apache.org/docs/apache-airflow/stable/installation/).

## Installing from PyPI

Install Airflow using the official PyPI package:

1.  Install just Airflow:

> Note: Only `pip` installation is currently officially supported.

While it is possible to install Airflow with tools like [Poetry](https://python-poetry.org) or
[pip-tools](https://pypi.org/project/pip-tools), they do not share the same workflow as
`pip` - especially when it comes to constraint vs. requirements management.
Installing via `Poetry` or `pip-tools` is not currently supported.

There are known issues with ``bazel`` that might lead to circular dependencies when using it to install
Airflow. Please switch to ``pip`` if you encounter such problems. ``Bazel`` community works on fixing
the problem in `this PR <https://github.com/bazelbuild/rules_python/pull/1166>`_ so it might be that
newer versions of ``bazel`` will handle it.

If you wish to install Airflow using those tools, you should use the constraint files and convert
them to the appropriate format and workflow that your tool requires.


```bash
pip install 'apache-airflow==3.0.3' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.3/constraints-3.10.txt"
```

2. Installing with extras (i.e., postgres, google)

```bash
pip install 'apache-airflow[postgres,google]==3.0.3' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.3/constraints-3.10.txt"
```

For information on installing provider distributions, check
[providers](http://airflow.apache.org/docs/apache-airflow-providers/index.html).

## Installation

For comprehensive instructions on setting up your local development environment and installing Apache Airflow, please refer to the [INSTALLING.md](INSTALLING.md) file.

## Official Source Code

*   Apache Airflow is an [Apache Software Foundation](https://www.apache.org) (ASF) project.
*   Official source code releases follow the [ASF Release Policy](https://www.apache.org/legal/release-policy.html)
*   Download from the [ASF Distribution Directory](https://downloads.apache.org/airflow).

## Convenience Packages

Airflow can also be installed via convenience methods:

*   [PyPI releases](https://pypi.org/project/apache-airflow/) using `pip`.
*   [Docker Images](https://hub.docker.com/r/apache/airflow).
*   [Tags in GitHub](https://github.com/apache/airflow/tags)

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

Airflow uses [SemVer](https://semver.org/) for versioning, with specific rules for core Airflow and providers:

*   **Airflow:**  Strict SemVer for the core package.
*   **Airflow Providers:**  SemVer applied to individual provider changes.

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

## Support for Python and Kubernetes versions

*   Airflow adheres to a defined policy for Python and Kubernetes support based on their official release schedules.
*   Support is dropped when versions reach EOL.

## Base OS support for reference Airflow images

*   Airflow provides container images with a stable Debian base OS.
*   Images include necessary packages, Python versions, database connectors, and popular providers.
*   Build your own custom images or use existing ones.

## Approach to dependencies of Airflow

Airflow has a policy on how dependencies are used, particularly regarding constraints.

## Contributing

Join the Airflow community!  Review the [contributors' guide](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst) for contribution details.

## Voting Policy

*   Commits need a +1 vote from a committer (not the author).
*   PMC and committer votes count for AIP voting.

## Who uses Apache Airflow?

See a list of organizations using Airflow [in the wild](https://github.com/apache/airflow/blob/main/INTHEWILD.md).

## Who maintains Apache Airflow?

Airflow is maintained by a community of [core committers/maintainers](https://people.apache.org/committers-by-project.html#airflow).

## What goes into the next release?

Information about releases can be found in the [What goes into the next release](dev/WHAT_GOES_INTO_THE_NEXT_RELEASE.md) document.

## Can I use the Apache Airflow logo in my presentation?

Yes, follow the Apache Foundation [trademark policies](https://www.apache.org/foundation/marks/#books) and the Apache Airflow [Brandbook](https://cwiki.apache.org/confluence/display/AIRFLOW/Brandbook).

## Links

*   [Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
*   [Chat](https://s.apache.org/airflow-slack)
*   [Community Information](https://airflow.apache.org/community/)

## Sponsors

<a href="https://astronomer.io"><img src="https://assets2.astronomer.io/logos/logoForLIGHTbackground.png" alt="astronomer.io" width="250px"></a>
<a href="https://aws.amazon.com/opensource/"><img src="https://github.com/apache/airflow/blob/main/providers/amazon/docs/integration-logos/AWS-Cloud-alt_light-bg@4x.png?raw=true" alt="AWS OpenSource" width="130px"></a>