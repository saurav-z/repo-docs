# Apache Airflow: Automate, Schedule, and Monitor Workflows

**Apache Airflow** is an open-source platform that empowers you to programmatically author, schedule, and monitor complex workflows. [Explore the original repo](https://github.com/apache/airflow)

## Key Features

*   **Dynamic**: Define pipelines in code for dynamic DAG generation and parameterization.
*   **Extensible**: Leverage a wide range of built-in operators and customize with your own.
*   **Flexible**: Utilize the Jinja templating engine for rich pipeline customization.
*   **Scalable:** Execute tasks on an array of workers while following specified dependencies.
*   **User-Friendly UI:** Visualize pipelines, monitor progress, and troubleshoot issues.

## Core Concepts

*   **DAGs (Directed Acyclic Graphs)**: Workflows are defined as code, making them maintainable, versionable, and testable.
*   **Scheduler**: Executes tasks based on dependencies and schedules.
*   **Operators**:  Building blocks for your workflows.
*   **XComs**:  Allows for metadata transfer between tasks.

## Requirements

Apache Airflow is tested with:

|            | Main version (dev)     | Stable version (3.0.2) |
|------------|------------------------|------------------------|
| Python     | 3.10, 3.11, 3.12       | 3.9, 3.10, 3.11, 3.12  |
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

Get started with Airflow using the [official documentation](https://airflow.apache.org/docs/apache-airflow/stable/) for installation, quickstart guides, and tutorials.

## Installing from PyPI

Install Apache Airflow from PyPI using the following command, specifying the version and constraint file for your Python version:

```bash
pip install 'apache-airflow==3.0.2' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.2/constraints-3.10.txt"
```

Install with extras (i.e., postgres, google):

```bash
pip install 'apache-airflow[postgres,google]==3.0.2' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.2/constraints-3.10.txt"
```

## Installation

For complete installation instructions, consult the [INSTALLING.md](INSTALLING.md) file.

## Official Source Code

Apache Airflow is an [Apache Software Foundation](https://www.apache.org) (ASF) project.  Official source code releases:

*   Follow the [ASF Release Policy](https://www.apache.org/legal/release-policy.html)
*   Can be downloaded from [the ASF Distribution Directory](https://downloads.apache.org/airflow)
*   Are cryptographically signed by the release manager
*   Are officially voted on by the PMC members during the
  [Release Approval Process](https://www.apache.org/legal/release-policy.html#release-approval)

## Convenience packages

There are other ways of installing and using Airflow. Those are "convenience" methods - they are
not "official releases" as stated by the `ASF Release Policy`, but they can be used by the users
who do not want to build the software themselves.

Those are - in the order of most common ways people install Airflow:

- [PyPI releases](https://pypi.org/project/apache-airflow/) to install Airflow using standard `pip` tool
- [Docker Images](https://hub.docker.com/r/apache/airflow) to install airflow via
  `docker` tool, use them in Kubernetes, Helm Charts, `docker-compose`, `docker swarm`, etc. You can
  read more about using, customizing, and extending the images in the
  [Latest docs](https://airflow.apache.org/docs/docker-stack/index.html), and
  learn details on the internals in the [images](https://airflow.apache.org/docs/docker-stack/index.html) document.
- [Tags in GitHub](https://github.com/apache/airflow/tags) to retrieve the git project sources that
  were used to generate official source packages via git

All those artifacts are not official releases, but they are prepared using officially released sources.
Some of those artifacts are "development" or "pre-release" ones, and they are clearly marked as such
following the ASF Policy.

## User Interface

The Airflow UI provides a comprehensive view of your workflows:

*   **DAGs**: Overview of all DAGs in your environment.
*   **Assets**: Overview of Assets with dependencies.
*   **Grid**: Grid representation of a DAG that spans across time.
*   **Graph**: Visualization of a DAG's dependencies and their current status for a specific run.
*   **Home**: Summary statistics of your Airflow environment.
*   **Backfill**: Backfilling a DAG for a specific date range.
*   **Code**: Quick way to view source code of a DAG.

## Semantic Versioning

Airflow follows [SemVer](https://semver.org/) for releases.

## Version Life Cycle

See the [Version Life Cycle](#version-life-cycle) section for supported versions.

## Support for Python and Kubernetes versions

Airflow follows specific rules for Python and Kubernetes version support, detailed in the [Support for Python and Kubernetes versions](#support-for-python-and-kubernetes-versions) section.

## Base OS support for reference Airflow images

Airflow's container images are built with a focus on stability, security, and performance. For more details on the underlying OS, see [Base OS support for reference Airflow images](#base-os-support-for-reference-airflow-images)

## Approach to dependencies of Airflow

Airflow's approach to dependencies is detailed in the [Approach to dependencies of Airflow](#approach-to-dependencies-of-airflow) section.

## Contributing

Contribute to Apache Airflow!  Refer to the [contributors' guide](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst) for details on how to get involved.

## Voting Policy

* Commits need a +1 vote from a committer who is not the author
* When we do AIP voting, both PMC member's and committer's `+1s` are considered a binding vote.

## Who uses Apache Airflow?

[Organizations using Apache Airflow](https://github.com/apache/airflow/blob/main/INTHEWILD.md)

## Who maintains Apache Airflow?

Airflow is maintained by a [community](https://github.com/apache/airflow/graphs/contributors) of contributors and [core committers/maintainers](https://people.apache.org/committers-by-project.html#airflow).

## What goes into the next release?

Understand the release process and milestone management at [What goes into the next release](#what-goes-into-the-next-release)

## Can I use the Apache Airflow logo in my presentation?

Yes, consult the [Apache Foundation trademark policies](https://www.apache.org/foundation/marks/#books) and the Apache Airflow [Brandbook](https://cwiki.apache.org/confluence/display/AIRFLOW/Brandbook).

## Links

*   [Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
*   [Chat](https://s.apache.org/airflow-slack)
*   [Community Information](https://airflow.apache.org/community/)

## Sponsors

<a href="https://astronomer.io"><img src="https://assets2.astronomer.io/logos/logoForLIGHTbackground.png" alt="astronomer.io" width="250px"></a>
<a href="https://aws.amazon.com/opensource/"><img src="https://github.com/apache/airflow/blob/main/providers/amazon/docs/integration-logos/AWS-Cloud-alt_light-bg@4x.png?raw=true" alt="AWS OpenSource" width="130px"></a>