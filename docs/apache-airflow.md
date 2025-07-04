# Apache Airflow: Orchestrate Your Workflows with Ease

**Apache Airflow** is an open-source platform that empowers you to programmatically author, schedule, and monitor complex workflows, making data pipelines more maintainable, versionable, testable, and collaborative.  [Explore the official repository!](https://github.com/apache/airflow)

## Key Features:

*   **Dynamic Workflows:** Define pipelines in code for dynamic DAG generation and parameterization.
*   **Extensible Architecture:** Leverage a wide range of built-in operators and easily extend the framework to fit your unique needs.
*   **Flexible Customization:** Customize your workflows using the powerful [Jinja](https://jinja.palletsprojects.com) templating engine.
*   **Rich User Interface:** Visualize pipelines, monitor progress, and troubleshoot issues with a user-friendly web UI.
*   **Scalable Scheduling:** Execute tasks on a scalable array of workers, managed by a robust scheduler.
*   **Idempotent Tasks:** Designed with idempotent tasks in mind, Airflow promotes data integrity.

## Project Focus

Airflow excels with mostly static, slowly changing workflows. Similar projects include [Luigi](https://github.com/spotify/luigi), [Oozie](https://oozie.apache.org/) and [Azkaban](https://azkaban.github.io/).  It's commonly used for data processing, and favors tasks that don't pass large data volumes between steps, often delegating heavy lifting to specialized external services.

## Requirements

Apache Airflow is tested with the following:

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

Get up and running with Airflow quickly by visiting the [official documentation](https://airflow.apache.org/docs/apache-airflow/stable/) for installation, tutorials, and more.

> Note: For the latest development branch documentation, visit [s.apache.org/airflow-docs](https://s.apache.org/airflow-docs/).

For Airflow Improvement Proposals (AIPs), check out the [Airflow Wiki](https://cwiki.apache.org/confluence/display/AIRFLOW/Airflow+Improvement+Proposals).  Documentation for dependent projects like provider distributions, Docker image, Helm Chart, you'll find it in [the documentation index](https://airflow.apache.org/docs/).

## Installing from PyPI

Install Apache Airflow using `pip`.

> Note: Only `pip` installation is currently officially supported.

```bash
pip install 'apache-airflow==3.0.2' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.2/constraints-3.10.txt"
```

To install with extras:

```bash
pip install 'apache-airflow[postgres,google]==3.0.2' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.2/constraints-3.10.txt"
```

For provider distribution installation details, see [providers](http://airflow.apache.org/docs/apache-airflow-providers/index.html).

## Installation

For comprehensive instructions on setting up your local development environment and installing Apache Airflow, please refer to the [INSTALLING.md](INSTALLING.md) file.

## Official Source Code

Apache Airflow is an [Apache Software Foundation](https://www.apache.org) (ASF) project, and all official source code releases:

*   Adhere to the [ASF Release Policy](https://www.apache.org/legal/release-policy.html)
*   Are downloadable from [the ASF Distribution Directory](https://downloads.apache.org/airflow)
*   Are cryptographically signed by the release manager
*   Undergo official voting by PMC members during the [Release Approval Process](https://www.apache.org/legal/release-policy.html#release-approval)

## Convenience Packages

Besides the official source code, you can also install and use Airflow through "convenience" methods such as:

*   [PyPI releases](https://pypi.org/project/apache-airflow/) with `pip`
*   [Docker Images](https://hub.docker.com/r/apache/airflow) to install airflow via
  `docker` tool, use them in Kubernetes, Helm Charts, `docker-compose`, `docker swarm`, etc. You can
  read more about using, customizing, and extending the images in the
  [Latest docs](https://airflow.apache.org/docs/docker-stack/index.html), and
  learn details on the internals in the [images](https://airflow.apache.org/docs/docker-stack/index.html) document.
*   [Tags in GitHub](https://github.com/apache/airflow/tags) to retrieve the git project sources that
  were used to generate official source packages via git

## User Interface

Airflow provides a rich user interface for monitoring and managing your workflows:

*   **DAGs:** Overview of all DAGs.

    ![DAGs](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/dags.png)

*   **Assets:** Overview of Assets with dependencies.

    ![Asset Dependencies](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/assets_graph.png)

*   **Grid:** Time-based DAG representation.

    ![Grid](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/grid.png)

*   **Graph:** DAG dependency and status visualization.

    ![Graph](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/graph.png)

*   **Home:** Summary of your Airflow environment.

    ![Home](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/home.png)

*   **Backfill:** Backfilling DAGs.

    ![Backfill](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/backfill.png)

*   **Code:** View DAG source code.

    ![Code](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/code.png)

## Semantic Versioning

Airflow uses [SemVer](https://semver.org/) principles for versioning.  Refer to the detailed rules in the documentation.

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

Limited support versions will be supported with security and critical bug fix only.
EOL versions will not get any fixes nor support.
We always recommend that all users run the latest available minor release for whatever major version is in use.
We **highly** recommend upgrading to the latest Airflow major release at the earliest convenient time and before the EOL date.

## Support for Python and Kubernetes Versions

Airflow follows a defined support policy for Python and Kubernetes versions, aligned with their official release cycles.

## Base OS Support for Reference Airflow Images

The Airflow Community provides pre-packaged container images based on a stable Debian OS, along with other useful components.

## Approach to Dependencies of Airflow

Airflow uses a `constraints` approach to manage dependencies, ensuring repeatable installations while allowing for flexibility in dependency versions.  Details on dependency management are available in the documentation.

### Approach for dependencies for Airflow Core

There are a few dependencies that we decided are important enough to upper-bound them by default, as they are
known to follow predictable versioning scheme, and we know that new versions of those are very likely to
bring breaking changes.

The important dependencies are:

* `SQLAlchemy`
* `Alembic`
* `Flask`
* `werkzeug`
* `celery`
* `kubernetes`

### Approach for dependencies in Airflow Providers and extras

Those `extras` and `providers` dependencies are maintained in `provider.yaml` of each provider.

By default, we should not upper-bound dependencies for providers, however each provider's maintainer
might decide to add additional limits (and justify them with comment).

## Contributing

Contribute to Apache Airflow by following the [contributors' guide](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst). Check out the [contribution quickstart](https://github.com/apache/airflow/blob/main/contributing-docs/03_contributors_quick_start.rst) for getting started.

## Voting Policy

*   Commits need a +1 vote from a committer who is not the author
*   When we do AIP voting, both PMC member's and committer's `+1s` are considered a binding vote.

## Who Uses Apache Airflow?

Find out [who's using Apache Airflow](https://github.com/apache/airflow/blob/main/INTHEWILD.md), with approximately 500 known organizations.

## Who Maintains Apache Airflow?

Airflow is maintained by the [community](https://github.com/apache/airflow/graphs/contributors), with key committers/maintainers responsible for reviewing and merging pull requests, and steering new feature requests. Learn more about [committer requirements](https://github.com/apache/airflow/blob/main/COMMITTERS.rst#guidelines-to-become-an-airflow-committer).

## What Goes into the Next Release?

The release process and how PRs and issues make their way into a release is explained in detail in this document under the [Semantic versioning](#semantic-versioning) chapter.
Also detailed is explained in the [What goes into the next release](dev/WHAT_GOES_INTO_THE_NEXT_RELEASE.md) document in the `dev` folder of the repository.

## Can I use the Apache Airflow logo in my presentation?

Yes, but be sure to adhere to the Apache Foundation [trademark policies](https://www.apache.org/foundation/marks/#books) and the Apache Airflow [Brandbook](https://cwiki.apache.org/confluence/display/AIRFLOW/Brandbook).

## Links

*   [Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
*   [Chat](https://s.apache.org/airflow-slack)
*   [Community Information](https://airflow.apache.org/community/)

## Sponsors

CI infrastructure for Apache Airflow is sponsored by:

<!-- Ordered by most recently "funded" -->

<a href="https://astronomer.io"><img src="https://assets2.astronomer.io/logos/logoForLIGHTbackground.png" alt="astronomer.io" width="250px"></a>
<a href="https://aws.amazon.com/opensource/"><img src="https://github.com/apache/airflow/blob/main/providers/amazon/docs/integration-logos/AWS-Cloud-alt_light-bg@4x.png?raw=true" alt="AWS OpenSource" width="130px"></a>