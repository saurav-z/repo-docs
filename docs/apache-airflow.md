# Apache Airflow: Orchestrate, Schedule, and Monitor Workflows with Ease

**Apache Airflow** is a powerful platform for programmatically authoring, scheduling, and monitoring workflows, enabling data engineers and data scientists to build robust and scalable data pipelines. **[Explore the Apache Airflow repository](https://github.com/apache/airflow)** to learn more!

## Key Features

*   **Dynamic Pipelines:** Define workflows as code for maintainability, version control, and collaboration.
*   **Extensible Architecture:** Leverage a rich set of built-in operators and easily extend Airflow to meet specific needs.
*   **Flexible Templating:** Customize workflows with the Jinja templating engine for advanced configurations.
*   **Rich User Interface:** Visualize pipeline runs, monitor progress, and troubleshoot issues through an intuitive UI.
*   **Scalable Scheduling:** Schedule and execute tasks on a distributed array of workers.

## Project Focus

Apache Airflow excels with mostly static and slowly changing workflows, providing clarity in the unit of work and continuity. It is well-suited for:

*   Data processing pipelines.
*   Idempotent tasks to avoid duplicate data.
*   Delegating high-volume, data-intensive tasks to external services.
*   Batch processing of real-time data.

## Principles

*   **Dynamic:** Pipelines defined in code enable dynamic DAG generation and parameterization.
*   **Extensible:** A wide range of built-in operators and the ability to extend it.
*   **Flexible:** Leverages Jinja templating engine for rich customizations.

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

Get up and running with Apache Airflow by visiting the official documentation for installation, getting started guides, and tutorials.

> Note: Documentation for the main branch (latest development branch) can be found on [s.apache.org/airflow-docs](https://s.apache.org/airflow-docs/).

## Installing from PyPI

Install Apache Airflow using `pip`:

```bash
pip install 'apache-airflow==3.0.5' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.5/constraints-3.10.txt"
```

Install with extras:

```bash
pip install 'apache-airflow[postgres,google]==3.0.5' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.5/constraints-3.10.txt"
```

For information on installing provider distributions, check
[providers](http://airflow.apache.org/docs/apache-airflow-providers/index.html).

## Installation

Comprehensive installation instructions are available in the [INSTALLING.md](INSTALLING.md) file.

## Official Source Code

Apache Airflow is an Apache Software Foundation (ASF) project.

*   Follows the [ASF Release Policy](https://www.apache.org/legal/release-policy.html)
*   Can be downloaded from [the ASF Distribution Directory](https://downloads.apache.org/airflow)
*   Are cryptographically signed by the release manager
*   Are officially voted on by the PMC members during the
    [Release Approval Process](https://www.apache.org/legal/release-policy.html#release-approval)

## Convenience Packages

*   [PyPI releases](https://pypi.org/project/apache-airflow/) to install Airflow using standard `pip` tool
*   [Docker Images](https://hub.docker.com/r/apache/airflow) to install airflow via
    `docker` tool, use them in Kubernetes, Helm Charts, `docker-compose`, `docker swarm`, etc. You can
    read more about using, customizing, and extending the images in the
    [Latest docs](https://airflow.apache.org/docs/docker-stack/index.html), and
    learn details on the internals in the [images](https://airflow.apache.org/docs/docker-stack/index.html) document.
*   [Tags in GitHub](https://github.com/apache/airflow/tags) to retrieve the git project sources that
    were used to generate official source packages via git

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

Airflow 2.0.0 and later uses SemVer.

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

## Support for Python and Kubernetes versions

Airflow follows a defined policy for Python and Kubernetes support, as described in the [Python Developer's Guide](https://devguide.python.org/#status-of-python-branches) and [Kubernetes version skew policy](https://kubernetes.io/docs/setup/release/version-skew-policy/).

## Base OS support for reference Airflow images

Airflow's container images are based on stable Debian OS.

## Approach to dependencies of Airflow

Airflow's dependency management uses constraints and upper bounds to ensure installation stability while allowing flexibility for users to upgrade dependencies.

### Approach for dependencies for Airflow Core

*   `SQLAlchemy`: Upper-bound to a specific MINOR version
*   `Alembic`: Upper-bound
*   `Flask`: Upper-bound to a specific MAJOR version
*   `werkzeug`: Upper-bound
*   `celery`: Upper-bound to the next MAJOR version
*   `kubernetes`: Upper-bound to the next MAJOR version

### Approach for dependencies in Airflow Providers and extras

Dependencies for providers and extras are maintained in `provider.yaml`. Dependencies are not typically upper-bound.

## Contributing

Contribute to Apache Airflow by following the [contributors' guide](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst).

## Voting Policy

*   Commits need a +1 vote from a committer who is not the author
*   When we do AIP voting, both PMC member's and committer's `+1s` are considered a binding vote.

## Who uses Apache Airflow?

Airflow is used by hundreds of organizations.

## Who maintains Apache Airflow?

The Airflow community maintains the project.

## What goes into the next release?

The release process is outlined [here](dev/WHAT_GOES_INTO_THE_NEXT_RELEASE.md).

## Can I use the Apache Airflow logo in my presentation?

Yes, following the Apache Foundation [trademark policies](https://www.apache.org/foundation/marks/#books) and the Apache Airflow [Brandbook](https://cwiki.apache.org/confluence/display/AIRFLOW/Brandbook).

## Links

*   [Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
*   [Chat](https://s.apache.org/airflow-slack)
*   [Community Information](https://airflow.apache.org/community/)

## Sponsors

*   <a href="https://astronomer.io"><img src="https://assets2.astronomer.io/logos/logoForLIGHTbackground.png" alt="astronomer.io" width="250px"></a>
*   <a href="https://aws.amazon.com/opensource/"><img src="https://github.com/apache/airflow/blob/main/providers/amazon/docs/integration-logos/AWS-Cloud-alt_light-bg@4x.png?raw=true" alt="AWS OpenSource" width="130px"></a>