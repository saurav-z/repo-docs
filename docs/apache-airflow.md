# Apache Airflow: The Open-Source Workflow Orchestration Platform

**Orchestrate complex workflows with ease using Apache Airflow, a platform that allows you to programmatically author, schedule, and monitor your data pipelines.** [Learn more at the original Apache Airflow repository](https://github.com/apache/airflow).

Airflow empowers data engineers and data scientists to build, manage, and monitor data pipelines efficiently. Define your workflows as code for enhanced maintainability, version control, testing, and collaboration.

**Key Features:**

*   **Dynamic Pipelines:** Define workflows using Python, enabling dynamic DAG generation and parameterization.
*   **Extensible Architecture:** Benefit from a wide array of built-in operators and a flexible framework for custom extensions.
*   **Flexible Templating:** Leverage the power of the Jinja templating engine for rich customizations.
*   **User-Friendly UI:** Visualize pipelines, monitor progress, and troubleshoot issues with an intuitive web interface.
*   **Scalability:** Airflow is designed to handle complex workflows with large datasets, supporting distributed execution and resource management.

## Project Focus

Airflow excels with mostly static and slowly changing workflows, where the DAG structure remains consistent run after run. This consistency enhances the unit of work and data integrity. Similar projects include [Luigi](https://github.com/spotify/luigi), [Oozie](https://oozie.apache.org/) and [Azkaban](https://azkaban.github.io/).

Airflow is commonly used to process data and tasks are ideally idempotent, ensuring data consistency. Delegate data-intensive tasks to specialized external services. While not a streaming solution, Airflow can process real-time data in batches.

## Principles

*   **Dynamic**: Pipelines are defined in code, enabling dynamic dag generation and parameterization.
*   **Extensible**: The Airflow framework includes a wide range of built-in operators and can be extended to fit your needs.
*   **Flexible**: Airflow leverages the [**Jinja**](https://jinja.palletsprojects.com) templating engine, allowing rich customizations.

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

Explore the official Airflow website documentation for guidance on:

*   [Installing Airflow](https://airflow.apache.org/docs/apache-airflow/stable/installation/)
*   [Getting Started](https://airflow.apache.org/docs/apache-airflow/stable/start.html)
*   [Tutorials](https://airflow.apache.org/docs/apache-airflow/stable/tutorial/)

For insights on Airflow Improvement Proposals (AIPs), see the [Airflow Wiki](https://cwiki.apache.org/confluence/display/AIRFLOW/Airflow+Improvement+Proposals).

Documentation for dependent projects like provider distributions, Docker image, Helm Chart, you'll find it in [the documentation index](https://airflow.apache.org/docs/).

## Installing from PyPI

Install Apache Airflow using `pip`:

```bash
pip install 'apache-airflow==3.0.6' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.6/constraints-3.10.txt"
```

Install with extras like postgres and google:

```bash
pip install 'apache-airflow[postgres,google]==3.0.6' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.6/constraints-3.10.txt"
```

For information on installing provider distributions, check
[providers](http://airflow.apache.org/docs/apache-airflow-providers/index.html).

## Installation

For complete setup and installation instructions, refer to the [INSTALLING.md](INSTALLING.md) file.

## Official Source Code

Apache Airflow is an [Apache Software Foundation](https://www.apache.org) (ASF) project, with official releases:

*   Adhering to the [ASF Release Policy](https://www.apache.org/legal/release-policy.html)
*   Available for download from [the ASF Distribution Directory](https://downloads.apache.org/airflow)
*   Cryptographically signed by release managers
*   Officially approved by PMC members during the [Release Approval Process](https://www.apache.org/legal/release-policy.html#release-approval)

The source packages released must be sufficient for a user to build and test the
release provided they have access to the appropriate platform and tools.

## Convenience Packages

Choose from alternative installation methods:

*   [PyPI releases](https://pypi.org/project/apache-airflow/) using `pip`
*   [Docker Images](https://hub.docker.com/r/apache/airflow) via `docker` (for Kubernetes, Helm Charts, etc.)
*   [GitHub Tags](https://github.com/apache/airflow/tags) for direct source code retrieval

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

Airflow 2.0.0 and later use [SemVer](https://semver.org/) for all package releases.

Specific rules apply:

*   **Airflow**: SemVer rules apply to core Airflow only.
*   **Airflow Providers**: SemVer applies to changes within each provider.
*   **Airflow Helm Chart**: SemVer applies to chart changes.
*   **Airflow API Clients**: Independent versioning.

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

*   Limited support includes security and critical bug fixes only.
*   EOL versions receive no further updates.
*   Upgrade to the latest minor release within your major version.
*   Upgrade to the latest major release before the EOL date.

## Support for Python and Kubernetes Versions

*   Follows official Python and Kubernetes release schedules.
*   Drop support upon EOL unless major cloud providers continue support.
*   New versions are supported after release and CI pipeline integration.

## Base OS Support for Reference Airflow Images

*   Container images provided by the Airflow Community.
*   Includes a stable Debian OS.
*   Contains supported Python versions, database clients, and popular providers.
*   Custom image builds are supported.

## Approach to Dependencies of Airflow

*   `constraints` ensures repeatable installations.
*   Dependencies are generally not upper-bound unless necessary.
*   Upper bounds are justified with comments and conditions.

### Approach for dependencies for Airflow Core

Important dependencies that are upper-bound by default are:

* `SQLAlchemy`: upper-bound to specific MINOR version
* `Alembic`: it is important to handle our migrations in predictable and performant way.
* `Flask`: limiting it to MAJOR version
* `werkzeug`:  It is tightly coupled with Flask libraries, and we should update them together
* `celery`: upper-bound it to the next MAJOR version.
* `kubernetes`: upper-bound it to the next MAJOR version.

### Approach for dependencies in Airflow Providers and extras

*   By default, we should not upper-bound dependencies for providers, however each provider's maintainer
    might decide to add additional limits (and justify them with comment).

## Contributing

Contribute to Apache Airflow:

*   Review the [contributors' guide](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst).
*   Use the [contribution quickstart](https://github.com/apache/airflow/blob/main/contributing-docs/03_contributors_quick_start.rst)
*   Official Docker (container) images for Apache Airflow are described in [images](https://github.com/apache/airflow/blob/main/dev/breeze/doc/ci/02_images.md).

## Voting Policy

*   Commits require a +1 vote from a committer who is not the author.
*   AIP voting considers PMC and committer +1s as binding.

## Who Uses Apache Airflow?

[Hundreds of organizations](https://github.com/apache/airflow/blob/main/INTHEWILD.md) use Apache Airflow. Add your organization via PR.

## Who Maintains Apache Airflow?

The [community](https://github.com/apache/airflow/graphs/contributors) develops Airflow, with [core committers/maintainers](https://people.apache.org/committers-by-project.html#airflow) responsible for reviewing PRs and feature requests. Become a maintainer by reviewing the Apache Airflow [committer requirements](https://github.com/apache/airflow/blob/main/COMMITTERS.rst#guidelines-to-become-an-airflow-committer).

## What goes into the next release?

*   Details are explained in detail in this README under the [Semantic versioning](#semantic-versioning) chapter, but
    in short, we have `MAJOR.MINOR.PATCH` versions of Airflow.
*   The release manager has the final decision about cherry-picking.

More context and **FAQ** about the patchlevel release can be found in the
[What goes into the next release](dev/WHAT_GOES_INTO_THE_NEXT_RELEASE.md) document in the `dev` folder of the
repository.

## Can I use the Apache Airflow logo in my presentation?

Yes!  Abide by the Apache Foundation [trademark policies](https://www.apache.org/foundation/marks/#books) and the Apache Airflow [Brandbook](https://cwiki.apache.org/confluence/display/AIRFLOW/Brandbook). Logos are found in [this repo](https://github.com/apache/airflow/tree/main/airflow-core/docs/img/logos/) and on the Apache Software Foundation [website](https://www.apache.org/logos/about.html).

## Links

*   [Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
*   [Chat](https://s.apache.org/airflow-slack)
*   [Community Information](https://airflow.apache.org/community/)

## Sponsors

CI infrastructure sponsored by:

<!-- Ordered by most recently "funded" -->

<a href="https://astronomer.io"><img src="https://assets2.astronomer.io/logos/logoForLIGHTbackground.png" alt="astronomer.io" width="250px"></a>
<a href="https://aws.amazon.com/opensource/"><img src="https://github.com/apache/airflow/blob/main/providers/amazon/docs/integration-logos/AWS-Cloud-alt_light-bg@4x.png?raw=true" alt="AWS OpenSource" width="130px"></a>