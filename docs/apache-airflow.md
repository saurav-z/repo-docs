# Apache Airflow: Orchestrate Your Workflows with Code

**Simplify your data pipelines and automate complex workflows with Apache Airflow, a robust, open-source platform that brings maintainability and scalability to your data engineering tasks.**  Get started today by visiting the [official Apache Airflow repository](https://github.com/apache/airflow).

## Key Features

*   **Dynamic:** Define pipelines as code, enabling dynamic DAG generation and parameterization.
*   **Extensible:** Leverage a wide array of built-in operators and easily customize the framework to meet your specific needs.
*   **Flexible:** Benefit from the power of the [**Jinja**](https://jinja.palletsprojects.com) templating engine, allowing rich customization.

## Core Functionality

*   **Workflow Orchestration:** Schedule and monitor workflows, manage dependencies, and execute tasks efficiently.
*   **User-Friendly Interface:** Visualize pipeline runs, track progress, and troubleshoot issues with an intuitive UI.
*   **Scalability:** Handle complex, data-intensive workflows with a distributed architecture.
*   **Idempotent Task Design:** Best practices emphasize idempotent tasks, promoting data integrity.
*   **XCom Support:** Tasks can pass metadata between each other with [XCom feature](https://airflow.apache.org/docs/apache-airflow/stable/concepts/xcoms.html).

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

Dive into the world of Airflow with ease! Begin by exploring the official documentation for the latest **stable** release to learn how to install, set up, and configure Airflow.

*   [Installation Guide](https://airflow.apache.org/docs/apache-airflow/stable/installation/)
*   [Getting Started](https://airflow.apache.org/docs/apache-airflow/stable/start.html)
*   [Tutorial](https://airflow.apache.org/docs/apache-airflow/stable/tutorial/)

> Note: For the latest development branch, visit [s.apache.org/airflow-docs](https://s.apache.org/airflow-docs/).

## Installation from PyPI

Install Apache Airflow using `pip` with the following command:

```bash
pip install 'apache-airflow==3.0.6' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.6/constraints-3.10.txt"
```

## Installation

For a complete guide to setting up your development environment and installing Apache Airflow, refer to the [INSTALLING.md](INSTALLING.md) file.

## Official Source Code

Apache Airflow is an Apache Software Foundation (ASF) project, and official source code releases follow these guidelines:

*   Adhere to the [ASF Release Policy](https://www.apache.org/legal/release-policy.html)
*   Available for download from [the ASF Distribution Directory](https://downloads.apache.org/airflow)
*   Cryptographically signed by the release manager
*   Approved by the PMC members during the [Release Approval Process](https://www.apache.org/legal/release-policy.html#release-approval)

## Convenience Packages

Airflow offers a few convenience packages, in order of common use:

*   [PyPI releases](https://pypi.org/project/apache-airflow/) to install Airflow using standard `pip` tool
*   [Docker Images](https://hub.docker.com/r/apache/airflow) to install airflow via
    `docker` tool, use them in Kubernetes, Helm Charts, `docker-compose`, `docker swarm`, etc. You can
    read more about using, customizing, and extending the images in the
    [Latest docs](https://airflow.apache.org/docs/docker-stack/index.html), and
    learn details on the internals in the [images](https://airflow.apache.org/docs/docker-stack/index.html) document.
*   [Tags in GitHub](https://github.com/apache/airflow/tags) to retrieve the git project sources that
    were used to generate official source packages via git

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

Airflow 2.0.0 and later versions strictly adhere to [SemVer](https://semver.org/).

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

## Support for Python and Kubernetes versions

*   Support for Python and Kubernetes versions is aligned with their respective EOL schedules.
*   New versions are added after official release and integration into our CI pipeline.

## Base OS Support for Reference Airflow Images

*   Based on the stable Debian OS.
*   Includes essential packages, Python versions, and database connectors.
*   Custom image building is supported.
*   Use the stable Debian releases until their end-of-regular support.

## Approach to Dependencies of Airflow

*   Dependencies are managed in `pyproject.toml` and `provider.yaml`.
*   Constraints ensure repeatable installations.
*   Some dependencies are upper-bound to specific MINOR version, due to their importance and the risk involved.
*   The `constraints` mechanism takes care about finding and upgrading all the non-upper bound dependencies
automatically (providing that all the tests pass).

## Contributing

Contribute to Apache Airflow by following the [contributors' guide](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst). If you can't wait to contribute, and want to get started asap, check out the [contribution quickstart](https://github.com/apache/airflow/blob/main/contributing-docs/03_contributors_quick_start.rst) here!

Official Docker (container) images for Apache Airflow are described in [images](https://github.com/apache/airflow/blob/main/dev/breeze/doc/ci/02_images.md).

## Voting Policy

*   Requires a +1 vote from a committer who is not the author.
*   PMC members' and committers' `+1s` are binding votes for AIP voting.

## Who Uses Apache Airflow?

A list of organizations using Apache Airflow can be found [here](https://github.com/apache/airflow/blob/main/INTHEWILD.md).

## Who Maintains Apache Airflow?

Apache Airflow is maintained by the [community](https://github.com/apache/airflow/graphs/contributors) and the [core committers/maintainers](https://people.apache.org/committers-by-project.html#airflow).

## What goes into the next release?

The release process follows the [Semver](https://semver.org/) versioning scheme as described in
[Airflow release process](https://airflow.apache.org/docs/apache-airflow/stable/release-process.html). More
details are explained in detail in this README under the [Semantic versioning](#semantic-versioning) chapter, but
in short, we have `MAJOR.MINOR.PATCH` versions of Airflow.

Generally we release `MINOR` versions of Airflow from a branch that is named after the MINOR version. For example
`2.7.*` releases are released from `v2-7-stable` branch, `2.8.*` releases are released from `v2-8-stable`
branch, etc.

1. Most of the time in our release cycle, when the branch for next `MINOR` branch is not yet created, all
PRs merged to `main` (unless they get reverted), will find their way to the next `MINOR` release. For example
if the last release is `2.7.3` and `v2-8-stable` branch is not created yet, the next `MINOR` release
is `2.8.0` and all PRs merged to main will be released in `2.8.0`. However, some PRs (bug-fixes and
doc-only changes) when merged, can be cherry-picked to current `MINOR` branch and released in the
next `PATCHLEVEL` release. For example, if `2.8.1` is already released and we are working on `2.9.0dev`,  then
marking a PR with `2.8.2` milestone means that it will be cherry-picked to `v2-8-test` branch and
released in `2.8.2rc1`, and eventually in `2.8.2`.

2. When we prepare for the next `MINOR` release, we cut new `v2-*-test` and `v2-*-stable` branch
and prepare `alpha`, `beta` releases for the next `MINOR` version, the PRs merged to main will still be
released in the next `MINOR` release until `rc` version is cut. This is happening because the `v2-*-test`
and `v2-*-stable` branches are rebased on top of main when next `beta` and `rc` releases are prepared.
For example, when we cut `2.10.0beta1` version, anything merged to main before `2.10.0rc1` is released,
will find its way to 2.10.0rc1.

3. Then, once we prepare the first RC candidate for the MINOR release, we stop moving the `v2-*-test` and
`v2-*-stable` branches and the PRs merged to main will be released in the next `MINOR` release.
However, some PRs (bug-fixes and doc-only changes) when merged, can be cherry-picked to current `MINOR`
branch and released in the next `PATCHLEVEL` release - for example when the last released version from `v2-10-stable`
branch is `2.10.0rc1`, some of the PRs from main can be marked as `2.10.0` milestone by committers,
the release manager will try to cherry-pick them into the release branch.
If successful, they will be released in `2.10.0rc2` and subsequently in `2.10.0`. This also applies to
subsequent `PATCHLEVEL` versions. When for example `2.10.1` is already released, marking a PR with
`2.10.2` milestone will mean that it will be cherry-picked to `v2-10-stable` branch and released in `2.10.2rc1`
and eventually in `2.10.2`.

The final decision about cherry-picking is made by the release manager.

Marking issues with a milestone is a bit different. Maintainers do not mark issues with a milestone usually,
normally they are only marked in PRs. If PR linked to the issue (and "fixing it") gets merged and released
in a specific version following the process described above, the issue will be automatically closed, no
milestone will be set for the issue, you need to check the PR that fixed the issue to see which version
it was released in.

However, sometimes maintainers mark issues with specific milestone, which means that the
issue is important to become a candidate to take a look when the release is being prepared. Since this is an
Open-Source project, where basically all contributors volunteer their time, there is no guarantee that specific
issue will be fixed in specific version. We do not want to hold the release because some issue is not fixed,
so in such case release manager will reassign such unfixed issues to the next milestone in case they are not
fixed in time for the current release. Therefore, the milestone for issue is more of an intent that it should be
looked at, than promise it will be fixed in the version.

More context and **FAQ** about the patchlevel release can be found in the
[What goes into the next release](dev/WHAT_GOES_INTO_THE_NEXT_RELEASE.md) document in the `dev` folder of the
repository.

## Can I use the Apache Airflow logo in my presentation?

Yes, but please adhere to the Apache Foundation [trademark policies](https://www.apache.org/foundation/marks/#books) and the Apache Airflow [Brandbook](https://cwiki.apache.org/confluence/display/AIRFLOW/Brandbook).

## Links

*   [Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
*   [Chat](https://s.apache.org/airflow-slack)
*   [Community Information](https://airflow.apache.org/community/)

## Sponsors

The CI infrastructure for Apache Airflow has been sponsored by:

<!-- Ordered by most recently "funded" -->

<a href="https://astronomer.io"><img src="https://assets2.astronomer.io/logos/logoForLIGHTbackground.png" alt="astronomer.io" width="250px"></a>
<a href="https://aws.amazon.com/opensource/"><img src="https://github.com/apache/airflow/blob/main/providers/amazon/docs/integration-logos/AWS-Cloud-alt_light-bg@4x.png?raw=true" alt="AWS OpenSource" width="130px"></a>