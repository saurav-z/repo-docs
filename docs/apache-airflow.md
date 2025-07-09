# Apache Airflow: Orchestrate Your Workflows with Code

**Apache Airflow** is a platform that allows you to programmatically author, schedule, and monitor your workflows, making them more maintainable, versionable, testable, and collaborative. 

[Get started with Apache Airflow](https://github.com/apache/airflow)

**Key Features:**

*   **Dynamic:** Define pipelines in code, enabling dynamic DAG generation and parameterization.
*   **Extensible:** Leverage a wide range of built-in operators and easily extend Airflow to fit your needs.
*   **Flexible:** Customize your workflows using the **Jinja** templating engine for rich control.
*   **Robust:** Schedule and monitor workflows with a user-friendly UI, easily visualizing pipelines, monitoring progress, and troubleshooting issues.

<picture width="500">
  <img
    src="https://github.com/apache/airflow/blob/19ebcac2395ef9a6b6ded3a2faa29dc960c1e635/docs/apache-airflow/img/logos/wordmark_1.png?raw=true"
    alt="Apache Airflow logo"
  />
</picture>

## Overview

[Apache Airflow](https://airflow.apache.org/docs/apache-airflow/stable/) (Airflow) is a leading open-source platform designed to programmatically author, schedule, and monitor complex workflows. By defining workflows as code (DAGs), Airflow promotes maintainability, version control, testability, and collaborative development.

## Why Choose Apache Airflow?

Airflow is ideal for automating and orchestrating data pipelines, ETL processes, and various other tasks. Its key strengths lie in:

*   **Workflow as Code:** Define your workflows using Python code, enabling version control, testing, and easier collaboration.
*   **Scheduling and Execution:** Schedule tasks using a robust scheduler and execute them on a distributed set of workers.
*   **Monitoring and Management:** Utilize a rich user interface to monitor pipeline progress, visualize dependencies, and troubleshoot issues.
*   **Extensibility:** Benefit from a wide range of built-in operators and easily extend Airflow to connect with various services and platforms.

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

Explore the following resources to get started:

*   **Installation:** [INSTALLING.md](INSTALLING.md)
*   **Documentation:** [Official Apache Airflow Documentation](https://airflow.apache.org/docs/apache-airflow/stable/) (latest stable release)
*   **Tutorials:** [Airflow Tutorial](https://airflow.apache.org/docs/apache-airflow/stable/tutorial/)
*   **Main branch (latest development):** [s.apache.org/airflow-docs](https://s.apache.org/airflow-docs/)
*   **Airflow Improvement Proposals (AIPs):** [Airflow Wiki](https://cwiki.apache.org/confluence/display/AIRFLOW/Airflow+Improvement+Proposals)

## Installing from PyPI

Install Apache Airflow using `pip`:

**Note:** Only `pip` installation is currently officially supported.

```bash
pip install 'apache-airflow==3.0.2' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.2/constraints-3.10.txt"
```

**Installing with Extras:**

```bash
pip install 'apache-airflow[postgres,google]==3.0.2' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.2/constraints-3.10.txt"
```

**For more detailed information on installing provider distributions, please see the following [providers](http://airflow.apache.org/docs/apache-airflow-providers/index.html)**

## Official Source Code

Apache Airflow is an [Apache Software Foundation](https://www.apache.org) (ASF) project.

*   **Official Releases:** Follow the [ASF Release Policy](https://www.apache.org/legal/release-policy.html)
*   **Download:** [ASF Distribution Directory](https://downloads.apache.org/airflow)
*   **Cryptographic Signatures:** Releases are cryptographically signed.
*   **Release Approval:** Officially voted on by PMC members.

## Convenience Packages

*   [PyPI releases](https://pypi.org/project/apache-airflow/)
*   [Docker Images](https://hub.docker.com/r/apache/airflow)
*   [Tags in GitHub](https://github.com/apache/airflow/tags)

## User Interface

Airflow offers a rich user interface for monitoring and managing workflows:

*   **DAGs:** Overview of all DAGs.

    ![DAGs](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/dags.png)
*   **Assets:** Overview of Assets with dependencies.

    ![Asset Dependencies](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/assets_graph.png)
*   **Grid:** Grid representation of a DAG that spans across time.

    ![Grid](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/grid.png)
*   **Graph:** Visualization of a DAG's dependencies and status.

    ![Graph](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/graph.png)
*   **Home:** Summary statistics of your Airflow environment.

    ![Home](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/home.png)
*   **Backfill:** Backfilling a DAG for a specific date range.

    ![Backfill](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/backfill.png)
*   **Code:** Quick way to view source code of a DAG.

    ![Code](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/code.png)

## Semantic Versioning

Apache Airflow follows [SemVer](https://semver.org/) for versioning, starting with Airflow 2.0.0.

## Version Life Cycle

Apache Airflow version life cycle:

<!-- This table is automatically updated by pre-commit scripts/ci/pre_commit/supported_versions.py -->
<!-- Beginning of auto-generated table -->

| Version   | Current Patch/Minor   | State     | First Release   | Limited Maintenance   | EOL/Terminated   |
|-----------|-----------------------|-----------|-----------------|-----------------------|------------------|
| 3         | 3.0.2                 | Supported | Apr 22, 2025    | TBD                   | TBD              |
| 2         | 2.11.0                | Supported | Dec 17, 2020    | Oct 22, 2025          | Apr 22, 2026     |
| 1.10      | 1.10.15               | EOL       | Aug 27, 2018    | Dec 17, 2020          | June 17, 2021    |
| 1.9       | 1.9.0                 | EOL       | Jan 03, 2018    | Aug 27, 2018          | Aug 27, 2018     |
| 1.8       | 1.8.2                 | EOL       | Mar 19, 2017    | Jan 03, 2018          | Jan 03, 2018     |
| 1.7       | 1.7.1.2               | EOL       | Mar 28, 2016    | Mar 19, 2017          | Mar 19, 2017     |

<!-- End of auto-generated table -->

## Support for Python and Kubernetes Versions

Airflow supports Python and Kubernetes versions based on their official release schedules.

## Base OS Support for Reference Airflow Images

The Airflow community provides container images:

*   Base OS (Debian)
*   Python versions
*   Database connectors
*   Predefined providers
*   Custom Image building

## Contributing

Contribute to Apache Airflow! See the [contributors' guide](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst) for details. For quick starts, try the [contribution quickstart](https://github.com/apache/airflow/blob/main/contributing-docs/03_contributors_quick_start.rst).

## Voting Policy

*   Commits require a +1 from a committer (other than the author).
*   AIP voting: PMC and committer +1s are binding.

## Who Uses Apache Airflow?

Apache Airflow is used by organizations worldwide.  See the list of users [in the wild](https://github.com/apache/airflow/blob/main/INTHEWILD.md).

## Who Maintains Apache Airflow?

Airflow is community-driven. The [core committers/maintainers](https://people.apache.org/committers-by-project.html#airflow) are responsible for reviews, merging PRs, and feature requests. Review the [committer requirements](https://github.com/apache/airflow/blob/main/COMMITTERS.rst#guidelines-to-become-an-airflow-committer) to become a maintainer.

## What goes into the next release?

The PRs will be merged to `main` branch until they get reverted, and it is planned to be released in the next MINOR version release. PRs (bug-fixes and doc-only changes) when merged, can be cherry-picked to current `MINOR` branch and released in the next `PATCHLEVEL` release. Please see the document in `dev` folder about the details of releasing process.

## Can I use the Apache Airflow logo in my presentation?

Yes, follow the Apache Foundation [trademark policies](https://www.apache.org/foundation/marks/#books) and the Apache Airflow [Brandbook](https://cwiki.apache.org/confluence/display/AIRFLOW/Brandbook).  Logos are found [here](https://github.com/apache/airflow/tree/main/airflow-core/docs/img/logos/) and on the Apache Software Foundation [website](https://www.apache.org/logos/about.html).

## Links

*   [Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
*   [Chat](https://s.apache.org/airflow-slack)
*   [Community Information](https://airflow.apache.org/community/)

## Sponsors

CI infrastructure for Apache Airflow is sponsored by:

<!-- Ordered by most recently "funded" -->

<a href="https://astronomer.io"><img src="https://assets2.astronomer.io/logos/logoForLIGHTbackground.png" alt="astronomer.io" width="250px"></a>
<a href="https://aws.amazon.com/opensource/"><img src="https://github.com/apache/airflow/blob/main/providers/amazon/docs/integration-logos/AWS-Cloud-alt_light-bg@4x.png?raw=true" alt="AWS OpenSource" width="130px"></a>