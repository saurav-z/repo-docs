# Apache Airflow: Orchestrate, Schedule, and Monitor Workflows at Scale

[Apache Airflow](https://github.com/apache/airflow) is an open-source platform that allows you to programmatically author, schedule, and monitor your workflows as directed acyclic graphs (DAGs).

**Key Features:**

*   **Workflow as Code:** Define workflows using Python, making them versionable, testable, and collaborative.
*   **Dynamic DAG Generation:** Easily create and parameterize workflows.
*   **Extensible:** Airflow supports a wide range of built-in operators and can be extended to fit your specific needs.
*   **Flexible:** Leverage Jinja templating for rich customization.
*   **Scalable:** Execute tasks on an array of workers with the Airflow scheduler.
*   **User-Friendly UI:** Visualize pipelines, monitor progress, and troubleshoot issues with a rich user interface.

**Get started with Apache Airflow and streamline your data pipelines!**

## Core Concepts

Apache Airflow excels at managing workflows that are relatively stable and gradually evolve. DAG structures are generally consistent from one run to the next, which clarifies the unit of work and its continuity. Airflow primarily focuses on orchestrating tasks, ideally idempotent operations. While suitable for batch processing of real-time data, it's not designed as a streaming solution.

## Project Focus

Airflow is ideal for workflows that are primarily static and change infrequently.

Other similar projects include [Luigi](https://github.com/spotify/luigi), [Oozie](https://oozie.apache.org/) and [Azkaban](https://azkaban.github.io/).

Airflow is commonly used to process data, but has the opinion that tasks should ideally be idempotent (i.e., results of the task will be the same, and will not create duplicated data in a destination system), and should not pass large quantities of data from one task to the next (though tasks can pass metadata using Airflow's [XCom feature](https://airflow.apache.org/docs/apache-airflow/stable/concepts/xcoms.html)). For high-volume, data-intensive tasks, a best practice is to delegate to external services specializing in that type of work.

Airflow is not a streaming solution, but it is often used to process real-time data, pulling data off streams in batches.

## Principles

-   **Dynamic:** Pipelines are defined in code, enabling dynamic dag generation and parameterization.
-   **Extensible:** The Airflow framework includes a wide range of built-in operators and can be extended to fit your needs.
-   **Flexible:** Airflow leverages the [**Jinja**](https://jinja.palletsprojects.com) templating engine, allowing rich customizations.

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

Explore the latest [stable](https://airflow.apache.org/docs/apache-airflow/stable/) documentation to learn about:

*   [Installation](https://airflow.apache.org/docs/apache-airflow/stable/installation/)
*   [Getting Started](https://airflow.apache.org/docs/apache-airflow/stable/start.html)
*   [Tutorial](https://airflow.apache.org/docs/apache-airflow/stable/tutorial/)

For documentation for the main branch (latest development branch), visit [s.apache.org/airflow-docs](https://s.apache.org/airflow-docs/).

For more information on Airflow Improvement Proposals (AIPs), visit
the [Airflow Wiki](https://cwiki.apache.org/confluence/display/AIRFLOW/Airflow+Improvement+Proposals).

Documentation for dependent projects like provider distributions, Docker image, Helm Chart, you'll find it in [the documentation index](https://airflow.apache.org/docs/).

## Installing from PyPI

Airflow is available on PyPI as the `apache-airflow` package. For reliable installations, it's recommended to use constraint files for your specific Python version.

1.  Install Airflow using constraints:

    ```bash
    pip install 'apache-airflow==3.0.3' \
     --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.3/constraints-3.10.txt"
    ```

2.  Install with extras (e.g., Postgres, Google):

    ```bash
    pip install 'apache-airflow[postgres,google]==3.0.3' \
     --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.3/constraints-3.10.txt"
    ```

For information on installing provider distributions, check
[providers](http://airflow.apache.org/docs/apache-airflow-providers/index.html).

## Installation

Detailed installation instructions are available in the [INSTALLING.md](INSTALLING.md) file.

## Official Source Code

Apache Airflow is an [Apache Software Foundation](https://www.apache.org) (ASF) project,
and our official source code releases:

- Follow the [ASF Release Policy](https://www.apache.org/legal/release-policy.html)
- Can be downloaded from [the ASF Distribution Directory](https://downloads.apache.org/airflow)
- Are cryptographically signed by the release manager
- Are officially voted on by the PMC members during the
  [Release Approval Process](https://www.apache.org/legal/release-policy.html#release-approval)

Following the ASF rules, the source packages released must be sufficient for a user to build and test the
release provided they have access to the appropriate platform and tools.

## Convenience Packages

*   [PyPI releases](https://pypi.org/project/apache-airflow/)
*   [Docker Images](https://hub.docker.com/r/apache/airflow)
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

Airflow uses SemVer for all packages. See the [Semantic Versioning](https://semver.org/) for more info.

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

We always recommend that all users run the latest available minor release for whatever major version is in use.
We **highly** recommend upgrading to the latest Airflow major release at the earliest convenient time and before the EOL date.

## Support for Python and Kubernetes versions

As of Airflow 2.0, we agreed to certain rules we follow for Python and Kubernetes support.
They are based on the official release schedule of Python and Kubernetes, nicely summarized in the
[Python Developer's Guide](https://devguide.python.org/#status-of-python-branches) and
[Kubernetes version skew policy](https://kubernetes.io/docs/setup/release/version-skew-policy/).

1.  We drop support for Python and Kubernetes versions when they reach EOL. Except for Kubernetes, a
    version stays supported by Airflow if two major cloud providers still provide support for it. We drop
    support for those EOL versions in main right after EOL date, and it is effectively removed when we release
    the first new MINOR (Or MAJOR if there is no new MINOR version) of Airflow. For example, for Python 3.10 it
    means that we will drop support in main right after 27.06.2023, and the first MAJOR or MINOR version of
    Airflow released after will not have it.

2.  We support a new version of Python/Kubernetes in main after they are officially released, as soon as we
    make them work in our CI pipeline (which might not be immediate due to dependencies catching up with
    new versions of Python mostly) we release new images/support in Airflow based on the working CI setup.

3.  This policy is best-effort which means there may be situations where we might terminate support earlier
    if circumstances require it.

## Base OS support for reference Airflow images

The Airflow Community provides conveniently packaged container images that are published whenever
we publish an Apache Airflow release. Those images contain:

* Base OS with necessary packages to install Airflow (stable Debian OS)
* Base Python installation in versions supported at the time of release for the MINOR version of
  Airflow released (so there could be different versions for 2.3 and 2.2 line for example)
* Libraries required to connect to supported Databases (again the set of databases supported depends
  on the MINOR version of Airflow)
* Predefined set of popular providers (for details see the [Dockerfile](https://raw.githubusercontent.com/apache/airflow/main/Dockerfile)).
* Possibility of building your own, custom image where the user can choose their own set of providers
  and libraries (see [Building the image](https://airflow.apache.org/docs/docker-stack/build.html))
* In the future Airflow might also support a "slim" version without providers nor database clients installed

The version of the base OS image is the stable version of Debian. Airflow supports using all currently active
stable versions - as soon as all Airflow dependencies support building, and we set up the CI pipeline for
building and testing the OS version. Approximately 6 months before the end-of-regular support of a
previous stable version of the OS, Airflow switches the images released to use the latest supported
version of the OS.

For example switch from ``Debian Bullseye`` to ``Debian Bookworm`` has been implemented
before 2.8.0 release in October 2023 and ``Debian Bookworm`` will be the only option supported as of
Airflow 2.10.0.

Users will continue to be able to build their images using stable Debian releases until the end of regular
support and building and verifying of the images happens in our CI but no unit tests were executed using
this image in the `main` branch.

## Approach to dependencies of Airflow

Airflow has a lot of dependencies - direct and transitive, also Airflow is both - library and application,
therefore our policies to dependencies has to include both - stability of installation of application,
but also ability to install newer version of dependencies for those users who develop DAGs. We developed
the approach where `constraints` are used to make sure airflow can be installed in a repeatable way, while
we do not limit our users to upgrade most of the dependencies. As a result we decided not to upper-bound
version of Airflow dependencies by default, unless we have good reasons to believe upper-bounding them is
needed because of importance of the dependency as well as risk it involves to upgrade specific dependency.
We also upper-bound the dependencies that we know cause problems.

The constraint mechanism of ours takes care about finding and upgrading all the non-upper bound dependencies
automatically (providing that all the tests pass). Our `main` build failures will indicate in case there
are versions of dependencies that break our tests - indicating that we should either upper-bind them or
that we should fix our code/tests to account for the upstream changes from those dependencies.

Whenever we upper-bound such a dependency, we should always comment why we are doing it - i.e. we should have
a good reason why dependency is upper-bound. And we should also mention what is the condition to remove the
binding.

### Approach for dependencies for Airflow Core

Those dependencies are maintained in ``pyproject.toml``.

There are few dependencies that we decided are important enough to upper-bound them by default, as they are
known to follow predictable versioning scheme, and we know that new versions of those are very likely to
bring breaking changes. We commit to regularly review and attempt to upgrade to the newer versions of
the dependencies as they are released, but this is manual process.

The important dependencies are:

*   `SQLAlchemy`: upper-bound to specific MINOR version (SQLAlchemy is known to remove deprecations and
    introduce breaking changes especially that support for different Databases varies and changes at
    various speed)
*   `Alembic`: it is important to handle our migrations in predictable and performant way. It is developed
    together with SQLAlchemy. Our experience with Alembic is that it very stable in MINOR version
*   `Flask`: We are using Flask as the back-bone of our web UI and API. We know major version of Flask
    are very likely to introduce breaking changes across those so limiting it to MAJOR version makes sense
*   `werkzeug`: the library is known to cause problems in new versions. It is tightly coupled with Flask
    libraries, and we should update them together
*   `celery`: Celery is a crucial component of Airflow as it used for CeleryExecutor (and similar). Celery
    [follows SemVer](https://docs.celeryq.dev/en/stable/contributing.html?highlight=semver#versions), so
    we should upper-bound it to the next MAJOR version. Also, when we bump the upper version of the library,
    we should make sure Celery Provider minimum Airflow version is updated.
*   `kubernetes`: Kubernetes is a crucial component of Airflow as it is used for the KubernetesExecutor
    (and similar). Kubernetes Python library [follows SemVer](https://github.com/kubernetes-client/python#compatibility),
    so we should upper-bound it to the next MAJOR version. Also, when we bump the upper version of the library,
    we should make sure Kubernetes Provider minimum Airflow version is updated.

### Approach for dependencies in Airflow Providers and extras

The main part of the Airflow is the Airflow Core, but the power of Airflow also comes from a number of
providers that extend the core functionality and are released separately, even if we keep them (for now)
in the same monorepo for convenience. You can read more about the providers in the
[Providers documentation](https://airflow.apache.org/docs/apache-airflow-providers/index.html). We also
have set of policies implemented for maintaining and releasing community-managed providers as well
as the approach for community vs. 3rd party providers in the [providers](https://github.com/apache/airflow/blob/main/PROVIDERS.rst) document.

Those `extras` and `providers` dependencies are maintained in `provider.yaml` of each provider.

By default, we should not upper-bound dependencies for providers, however each provider's maintainer
might decide to add additional limits (and justify them with comment).

## Contributing

Contribute to Apache Airflow! See our [contributors' guide](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst) for details.

## Voting Policy

*   Commits need a +1 vote from a committer who is not the author
*   When we do AIP voting, both PMC member's and committer's `+1s` are considered a binding vote.

## Who uses Apache Airflow?

See [INTHEWILD.md](https://github.com/apache/airflow/blob/main/INTHEWILD.md) for organizations using Airflow.

## Who maintains Apache Airflow?

Airflow is maintained by a [community](https://github.com/apache/airflow/graphs/contributors), led by the [core committers/maintainers](https://people.apache.org/committers-by-project.html#airflow).

If you would like to become a maintainer, please review the Apache Airflow
[committer requirements](https://github.com/apache/airflow/blob/main/COMMITTERS.rst#guidelines-to-become-an-airflow-committer).

## What goes into the next release?

Often you will see an issue that is assigned to specific milestone with Airflow version, or a PR that gets merged
to the main branch and you might wonder which release the merged PR(s) will be released in or which release the fixed
issues will be in. The answer to this is as usual - it depends on various scenarios. The answer is different for PRs and Issues.

To add a bit of context, we are following the [Semver](https://semver.org/) versioning scheme as described in
[Airflow release process](https://airflow.apache.org/docs/apache-airflow/stable/release-process.html). More
details are explained in detail in this README under the [Semantic versioning](#semantic-versioning) chapter, but
in short, we have `MAJOR.MINOR.PATCH` versions of Airflow.

*   `MAJOR` version is incremented in case of breaking changes
*   `MINOR` version is incremented when there are new features added
*   `PATCH` version is incremented when there are only bug-fixes and doc-only changes

Generally we release `MINOR` versions of Airflow from a branch that is named after the MINOR version. For example
`2.7.*` releases are released from `v2-7-stable` branch, `2.8.*` releases are released from `v2-8-stable`
branch, etc.

1.  Most of the time in our release cycle, when the branch for next `MINOR` branch is not yet created, all
    PRs merged to `main` (unless they get reverted), will find their way to the next `MINOR` release. For example
    if the last release is `2.7.3` and `v2-8-stable` branch is not created yet, the next `MINOR` release
    is `2.8.0` and all PRs merged to main will be released in `2.8.0`. However, some PRs (bug-fixes and
    doc-only changes) when merged, can be cherry-picked to current `MINOR` branch and released in the
    next `PATCHLEVEL` release. For example, if `2.8.1` is already released and we are working on `2.9.0dev`,  then
    marking a PR with `2.8.2` milestone means that it will be cherry-picked to `v2-8-test` branch and
    released in `2.8.2rc1`, and eventually in `2.8.2`.

2.  When we prepare for the next `MINOR` release, we cut new `v2-*-test` and `v2-*-stable` branch
    and prepare `alpha`, `beta` releases for the next `MINOR` version, the PRs merged to main will still be
    released in the next `MINOR` release until `rc` version is cut. This is happening because the `v2-*-test`
    and `v2-*-stable` branches are rebased on top of main when next `beta` and `rc` releases are prepared.
    For example, when we cut `2.10.0beta1` version, anything merged to main before `2.10.0rc1` is released,
    will find its way to 2.10.0rc1.

3.  Then, once we prepare the first RC candidate for the MINOR release, we stop moving the `v2-*-test` and
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

Yes, but follow the Apache Foundation [trademark policies](https://www.apache.org/foundation/marks/#books) and the Apache Airflow [Brandbook](https://cwiki.apache.org/confluence/display/AIRFLOW/Brandbook). Logos are in the [official repo](https://github.com/apache/airflow/tree/main/airflow-core/docs/img/logos/) and on the ASF [website](https://www.apache.org/logos/about.html).

## Links

*   [Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
*   [Chat](https://s.apache.org/airflow-slack)
*   [Community Information](https://airflow.apache.org/community/)

## Sponsors

<a href="https://astronomer.io"><img src="https://assets2.astronomer.io/logos/logoForLIGHTbackground.png" alt="astronomer.io" width="250px"></a>
<a href="https://aws.amazon.com/opensource/"><img src="https://github.com/apache/airflow/blob/main/providers/amazon/docs/integration-logos/AWS-Cloud-alt_light-bg@4x.png?raw=true" alt="AWS OpenSource" width="130px"></a>