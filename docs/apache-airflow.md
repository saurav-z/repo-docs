# Apache Airflow: Orchestrate Your Workflows with Code

**Apache Airflow is a powerful platform for programmatically authoring, scheduling, and monitoring your workflows.** [Explore the official repository](https://github.com/apache/airflow)

<picture width="500">
  <img
    src="https://github.com/apache/airflow/blob/19ebcac2395ef9a6b6ded3a2faa29dc960c1e635/docs/apache-airflow/img/logos/wordmark_1.png?raw=true"
    alt="Apache Airflow logo"
  />
</picture>

Apache Airflow (or simply Airflow) is a leading open-source workflow management platform. It enables you to define, schedule, and monitor complex data pipelines and other workflows as code, making them more maintainable, versionable, testable, and collaborative.

## Key Features of Apache Airflow:

*   **Dynamic Pipelines:** Define workflows in Python code for dynamic DAG generation and parameterization.
*   **Extensible Architecture:** Integrate with a wide array of existing tools and services via a rich library of built-in operators.
*   **Flexible Templating:** Utilize Jinja templating for highly customizable workflows.
*   **Scalable Scheduling & Execution:** Schedule tasks with robust command-line utilities for complex operations.
*   **Intuitive UI:** Visualize pipelines, monitor progress, and troubleshoot issues with a user-friendly interface.

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

Visit the official Airflow website documentation (latest **stable** release) for help with
[installing Airflow](https://airflow.apache.org/docs/apache-airflow/stable/installation/),
[getting started](https://airflow.apache.org/docs/apache-airflow/stable/start.html), or walking
through a more complete [tutorial](https://airflow.apache.org/docs/apache-airflow/stable/tutorial/).

> Note: If you're looking for documentation for the main branch (latest development branch): you can find it on [s.apache.org/airflow-docs](https://s.apache.org/airflow-docs/).

For more information on Airflow Improvement Proposals (AIPs), visit
the [Airflow Wiki](https://cwiki.apache.org/confluence/display/AIRFLOW/Airflow+Improvement+Proposals).

Documentation for dependent projects like provider distributions, Docker image, Helm Chart, you'll find it in [the documentation index](https://airflow.apache.org/docs/).

## Installing from PyPI

We publish Apache Airflow as `apache-airflow` package in PyPI. Installing it however might be sometimes tricky
because Airflow is a bit of both a library and application. Libraries usually keep their dependencies open, and
applications usually pin them, but we should do neither and both simultaneously. We decided to keep
our dependencies as open as possible (in `pyproject.toml`) so users can install different versions of libraries
if needed. This means that `pip install apache-airflow` will not work from time to time or will
produce unusable Airflow installation.

To have repeatable installation, however, we keep a set of "known-to-be-working" constraint
files in the orphan `constraints-main` and `constraints-2-0` branches. We keep those "known-to-be-working"
constraints files separately per major/minor Python version.
You can use them as constraint files when installing Airflow from PyPI. Note that you have to specify
correct Airflow tag/version/branch and Python versions in the URL.

1. Installing just Airflow:

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
pip install 'apache-airflow==3.0.2' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.2/constraints-3.10.txt"
```

2. Installing with extras (i.e., postgres, google)

```bash
pip install 'apache-airflow[postgres,google]==3.0.2' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.2/constraints-3.10.txt"
```

For information on installing provider distributions, check
[providers](http://airflow.apache.org/docs/apache-airflow-providers/index.html).

## Installation

For comprehensive instructions on setting up your local development environment and installing Apache Airflow, please refer to the [INSTALLING.md](INSTALLING.md) file.

## Official Source Code

Apache Airflow is an [Apache Software Foundation](https://www.apache.org) (ASF) project,
and our official source code releases:

*   Follow the [ASF Release Policy](https://www.apache.org/legal/release-policy.html)
*   Can be downloaded from [the ASF Distribution Directory](https://downloads.apache.org/airflow)
*   Are cryptographically signed by the release manager
*   Are officially voted on by the PMC members during the
    [Release Approval Process](https://www.apache.org/legal/release-policy.html#release-approval)

Following the ASF rules, the source packages released must be sufficient for a user to build and test the
release provided they have access to the appropriate platform and tools.

## Convenience Packages

There are other ways of installing and using Airflow. Those are "convenience" methods - they are
not "official releases" as stated by the `ASF Release Policy`, but they can be used by the users
who do not want to build the software themselves.

Those are - in the order of most common ways people install Airflow:

*   [PyPI releases](https://pypi.org/project/apache-airflow/) to install Airflow using standard `pip` tool
*   [Docker Images](https://hub.docker.com/r/apache/airflow) to install airflow via
    `docker` tool, use them in Kubernetes, Helm Charts, `docker-compose`, `docker swarm`, etc. You can
    read more about using, customizing, and extending the images in the
    [Latest docs](https://airflow.apache.org/docs/docker-stack/index.html), and
    learn details on the internals in the [images](https://airflow.apache.org/docs/docker-stack/index.html) document.
*   [Tags in GitHub](https://github.com/apache/airflow/tags) to retrieve the git project sources that
    were used to generate official source packages via git

All those artifacts are not official releases, but they are prepared using officially released sources.
Some of those artifacts are "development" or "pre-release" ones, and they are clearly marked as such
following the ASF Policy.

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

As of Airflow 2.0.0, we support a strict [SemVer](https://semver.org/) approach for all packages released.

There are few specific rules that we agreed to that define details of versioning of the different
packages:

*   **Airflow**: SemVer rules apply to core airflow only (excludes any changes to providers).
    Changing limits for versions of Airflow dependencies is not a breaking change on its own.
*   **Airflow Providers**: SemVer rules apply to changes in the particular provider's code only.
    SemVer MAJOR and MINOR versions for the packages are independent of the Airflow version.
    For example, `google 4.1.0` and `amazon 3.0.3` providers can happily be installed
    with `Airflow 2.1.2`. If there are limits of cross-dependencies between providers and Airflow packages,
    they are present in providers as `install_requires` limitations. We aim to keep backwards
    compatibility of providers with all previously released Airflow 2 versions but
    there will sometimes be breaking changes that might make some, or all
    providers, have minimum Airflow version specified.
*   **Airflow Helm Chart**: SemVer rules apply to changes in the chart only. SemVer MAJOR and MINOR
    versions for the chart are independent of the Airflow version. We aim to keep backwards
    compatibility of the Helm Chart with all released Airflow 2 versions, but some new features might
    only work starting from specific Airflow releases. We might however limit the Helm
    Chart to depend on minimal Airflow version.
*   **Airflow API clients**: Their versioning is independent from Airflow versions. They follow their own
    SemVer rules for breaking changes and new features - which for example allows to change the way we generate
    the clients.

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

## Base OS Support for Reference Airflow Images

The Airflow Community provides conveniently packaged container images that are published whenever
we publish an Apache Airflow release. Those images contain:

*   Base OS with necessary packages to install Airflow (stable Debian OS)
*   Base Python installation in versions supported at the time of release for the MINOR version of
    Airflow released (so there could be different versions for 2.3 and 2.2 line for example)
*   Libraries required to connect to supported Databases (again the set of databases supported depends
    on the MINOR version of Airflow)
*   Predefined set of popular providers (for details see the [Dockerfile](https://raw.githubusercontent.com/apache/airflow/main/Dockerfile)).
*   Possibility of building your own, custom image where the user can choose their own set of providers
    and libraries (see [Building the image](https://airflow.apache.org/docs/docker-stack/build.html))
*   In the future Airflow might also support a "slim" version without providers nor database clients installed

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

## Approach to Dependencies of Airflow

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

### Approach for Dependencies for Airflow Core

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

### Approach for Dependencies in Airflow Providers and Extras

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

Help improve Apache Airflow!  Consult the [contributors' guide](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst) for instructions.  Or, jump right in with the [contribution quickstart](https://github.com/apache/airflow/blob/main/contributing-docs/03_contributors_quick_start.rst).

## Voting Policy

*   Commits need a +1 vote from a committer who is not the author
*   When we do AIP voting, both PMC member's and committer's `+1s` are considered a binding vote.

## Who Uses Apache Airflow?

Over 500 organizations are known to use Apache Airflow; add your organization to the list! See [INTHEWILD.md](https://github.com/apache/airflow/blob/main/INTHEWILD.md).

## Who Maintains Apache Airflow?

Airflow is a community project. The [core committers/maintainers](https://people.apache.org/committers-by-project.html#airflow) are responsible for merging PRs and steering new feature requests.

## What Goes Into the Next Release?

The issues and PRs included in the next release are determined by the release managers; see the
[What goes into the next release](dev/WHAT_GOES_INTO_THE_NEXT_RELEASE.md) document in the `dev` folder.

## Can I Use the Apache Airflow Logo in My Presentation?

Yes, please! Follow the Apache Foundation [trademark policies](https://www.apache.org/foundation/marks/#books) and the Apache Airflow [Brandbook](https://cwiki.apache.org/confluence/display/AIRFLOW/Brandbook).

## Links

*   [Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
*   [Chat](https://s.apache.org/airflow-slack)
*   [Community Information](https://airflow.apache.org/community/)

## Sponsors

The CI infrastructure for Apache Airflow is sponsored by:

<!-- Ordered by most recently "funded" -->

<a href="https://astronomer.io"><img src="https://assets2.astronomer.io/logos/logoForLIGHTbackground.png" alt="astronomer.io" width="250px"></a>
<a href="https://aws.amazon.com/opensource/"><img src="https://github.com/apache/airflow/blob/main/providers/amazon/docs/integration-logos/AWS-Cloud-alt_light-bg@4x.png?raw=true" alt="AWS OpenSource" width="130px"></a>