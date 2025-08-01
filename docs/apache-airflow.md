# Apache Airflow: Orchestrate, Schedule, and Monitor Workflows

**Apache Airflow** is a powerful platform that allows you to programmatically author, schedule, and monitor your workflows, making them more maintainable and efficient.  Get started with the official Apache Airflow project [here](https://github.com/apache/airflow)!

## Key Features

*   **Dynamic:** Define workflows as code for dynamic generation and parameterization.
*   **Extensible:** Utilize a wide range of built-in operators and customize to your needs.
*   **Flexible:** Leverage Jinja templating for rich workflow customizations.
*   **Scalable:** Execute tasks on a distributed array of workers.
*   **User-Friendly:** Monitor progress and troubleshoot issues using a rich web UI.

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

Access the official [Airflow documentation](https://airflow.apache.org/docs/apache-airflow/stable/) for installation, setup, and tutorials.

## Installation

For detailed installation instructions, consult the [INSTALLING.md](INSTALLING.md) file.

## Installing from PyPI

Install Airflow using pip with constraints for repeatable installations.

1.  Installing just Airflow:

```bash
pip install 'apache-airflow==3.0.3' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.3/constraints-3.10.txt"
```

2.  Installing with extras (i.e., postgres, google):

```bash
pip install 'apache-airflow[postgres,google]==3.0.3' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.3/constraints-3.10.txt"
```

## Official Source Code

Airflow is an Apache Software Foundation project.  Official source code releases:

*   Follow the [ASF Release Policy](https://www.apache.org/legal/release-policy.html)
*   Available from the [ASF Distribution Directory](https://downloads.apache.org/airflow)
*   Cryptographically signed by the release manager
*   Officially voted on by the PMC members during the [Release Approval Process](https://www.apache.org/legal/release-policy.html#release-approval)

## Convenience Packages

You can install Airflow with:

*   [PyPI releases](https://pypi.org/project/apache-airflow/)
*   [Docker Images](https://hub.docker.com/r/apache/airflow)
*   [Tags in GitHub](https://github.com/apache/airflow/tags)

## User Interface

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

Airflow 2.0.0+ uses [SemVer](https://semver.org/).

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

Airflow follows a defined policy for Python and Kubernetes version support based on their official release schedules.

## Base OS support for reference Airflow images

Airflow provides container images with a Debian base OS, and supported database clients.

## Approach to dependencies of Airflow

Airflow uses constraints and upper-bounding of dependencies to ensure repeatable installations.

## Contributing

Contribute to Airflow by following the [contributors' guide](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst).

## Voting Policy

*   Commits need a +1 vote from a committer who is not the author
*   AIP voting require both PMC and Committer's +1 vote.

## Who uses Apache Airflow?

Find out which organizations are using Apache Airflow [in the wild](https://github.com/apache/airflow/blob/main/INTHEWILD.md).

## Who maintains Apache Airflow?

Airflow is a community-driven project. [Core committers/maintainers](https://people.apache.org/committers-by-project.html#airflow) are responsible for reviewing and merging PRs.

## What goes into the next release?

Details about the release process are available at:  [dev/WHAT_GOES_INTO_THE_NEXT_RELEASE.md](dev/WHAT_GOES_INTO_THE_NEXT_RELEASE.md)

## Can I use the Apache Airflow logo in my presentation?

Yes! Follow the Apache Foundation [trademark policies](https://www.apache.org/foundation/marks/#books) and the Apache Airflow [Brandbook](https://cwiki.apache.org/confluence/display/AIRFLOW/Brandbook).

## Links

*   [Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
*   [Chat](https://s.apache.org/airflow-slack)
*   [Community Information](https://airflow.apache.org/community/)

## Sponsors

<a href="https://astronomer.io"><img src="https://assets2.astronomer.io/logos/logoForLIGHTbackground.png" alt="astronomer.io" width="250px"></a>
<a href="https://aws.amazon.com/opensource/"><img src="https://github.com/apache/airflow/blob/main/providers/amazon/docs/integration-logos/AWS-Cloud-alt_light-bg@4x.png?raw=true" alt="AWS OpenSource" width="130px"></a>