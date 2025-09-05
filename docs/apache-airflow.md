<!-- START Apache Airflow, please keep comment here to allow auto update of PyPI readme.md -->
# Apache Airflow: Automate and Monitor Your Workflows

**Apache Airflow is a powerful platform for programmatically authoring, scheduling, and monitoring data pipelines and workflows.** [Explore the original repository](https://github.com/apache/airflow) to learn more.

## Key Features

*   **Dynamic Workflows:** Define pipelines as code for flexibility and maintainability.
*   **Extensibility:** Easily extend Airflow with custom operators and integrations.
*   **Rich User Interface:** Visualize, monitor, and troubleshoot pipelines with ease.
*   **Scalable:** Designed to handle complex workflows with a focus on idempotency and delegating to specialized services.
*   **Community Driven:**  Benefit from a vibrant community of contributors and users.

## Project Focus

Airflow excels at orchestrating workflows, particularly those that are mostly static and slowly changing, with a focus on idempotent tasks and delegation to external services for high-volume data processing.

*   **Dynamic:** Pipelines are defined in code, enabling dynamic dag generation and parameterization.
*   **Extensible:** The Airflow framework includes a wide range of built-in operators and can be extended to fit your needs.
*   **Flexible:** Airflow leverages the [**Jinja**](https://jinja.palletsprojects.com) templating engine, allowing rich customizations.

<!-- START Requirements, please keep comment here to allow auto update of PyPI readme.md -->
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

<!-- END Requirements, please keep comment here to allow auto update of PyPI readme.md -->
<!-- START Getting started, please keep comment here to allow auto update of PyPI readme.md -->
## Getting Started

*   **Installation:**  Refer to the [INSTALLING.md](INSTALLING.md) file for detailed setup instructions.
*   **Documentation:** Explore the official Airflow documentation for the [latest stable release](https://airflow.apache.org/docs/apache-airflow/stable/) and  [getting started guides](https://airflow.apache.org/docs/apache-airflow/stable/start.html) and [tutorial](https://airflow.apache.org/docs/apache-airflow/stable/tutorial/).
*   **Development Branch Documentation**: For more information on Airflow Improvement Proposals (AIPs), visit
    the [Airflow Wiki](https://cwiki.apache.org/confluence/display/AIRFLOW/Airflow+Improvement+Proposals).
*   **Dependent Projects Documentation:** Documentation for dependent projects like provider distributions, Docker image, Helm Chart, you'll find it in [the documentation index](https://airflow.apache.org/docs/).

> Note: If you're looking for documentation for the main branch (latest development branch): you can find it on [s.apache.org/airflow-docs](https://s.apache.org/airflow-docs/).

<!-- END Getting started, please keep comment here to allow auto update of PyPI readme.md -->
<!-- START Installing from PyPI, please keep comment here to allow auto update of PyPI readme.md -->

## Installing from PyPI

Install Apache Airflow using `pip`.

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
pip install 'apache-airflow==3.0.6' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.6/constraints-3.10.txt"
```

2. Installing with extras (i.e., postgres, google)

```bash
pip install 'apache-airflow[postgres,google]==3.0.6' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.6/constraints-3.10.txt"
```

For information on installing provider distributions, check
[providers](http://airflow.apache.org/docs/apache-airflow-providers/index.html).

<!-- END Installing from PyPI, please keep comment here to allow auto update of PyPI readme.md -->

## Installation

For comprehensive instructions on setting up your local development environment and installing Apache Airflow, please refer to the [INSTALLING.md](INSTALLING.md) file.

<!-- START Official source code, please keep comment here to allow auto update of PyPI readme.md -->
## Official Source Code

Apache Airflow is an [Apache Software Foundation](https://www.apache.org) (ASF) project. The official source code releases:

*   Adhere to the [ASF Release Policy](https://www.apache.org/legal/release-policy.html).
*   Are available for download from [the ASF Distribution Directory](https://downloads.apache.org/airflow).
*   Are cryptographically signed by the release manager.
*   Undergo an official vote by PMC members during the [Release Approval Process](https://www.apache.org/legal/release-policy.html#release-approval).

Following the ASF rules, the source packages released must be sufficient for a user to build and test the
release provided they have access to the appropriate platform and tools.

<!-- END Official source code, please keep comment here to allow auto update of PyPI readme.md -->
## Convenience Packages

Airflow offers convenience packages. These are methods to install Airflow, but not "official releases" as stated by the `ASF Release Policy`, but they can be used by the users who do not want to build the software themselves.

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

Airflow provides a rich user interface for monitoring and managing your workflows:

*   **DAGs:** Overview of all DAGs.
  ![DAGs](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/dags.png)

*   **Assets:** Overview of Assets with dependencies.
  ![Asset Dependencies](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/assets_graph.png)

*   **Grid:** Time-based representation of DAGs.
  ![Grid](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/grid.png)

*   **Graph:** Visualizes dependencies and statuses.
  ![Graph](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/graph.png)

*   **Home:** Summary statistics of your Airflow environment.
  ![Home](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/home.png)

*   **Backfill:** Backfilling a DAG for a specific date range.
  ![Backfill](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/backfill.png)

*   **Code:** View DAG source code.
  ![Code](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/code.png)

## Semantic Versioning

Airflow 2.0.0 and later use [SemVer](https://semver.org/) for all released packages.

Rules for versioning of the different packages:

*   **Airflow**: SemVer rules apply to core airflow only (excludes any changes to providers).
    Changing limits for versions of Airflow dependencies is not a breaking change on its own.
*   **Airflow Providers**: SemVer rules apply to changes in the particular provider's code only.
    SemVer MAJOR and MINOR versions for the packages are independent of the Airflow version.
    For example, `google 4.1.0` and `amazon 3.0.6` providers can happily be installed
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

*   **Supported:** Actively maintained and receiving bug fixes and security updates.
*   **Limited Maintenance:** Only security and critical bug fixes are provided.
*   **EOL/Terminated:** No longer supported, receive no updates.
*   **Recommendation:** Upgrade to the latest minor release within your major version, and to the latest major release before the EOL date.

## Support for Python and Kubernetes Versions

Airflow follows these rules for Python and Kubernetes version support, based on their respective EOL policies:

1.  Drop support for EOL versions immediately after their official EOL dates, unless two major cloud providers still support them.
2.  Support new Python/Kubernetes versions in `main` after their official releases, as soon as they work in our CI pipeline.

## Base OS Support for Reference Airflow Images

Airflow provides container images with Debian as the base OS. The images include:

*   Debian stable with packages to install Airflow
*   Supported Python versions
*   Libraries for database connections
*   Predefined popular providers
*   Custom image builds

The base OS will transition to the latest stable Debian release before the end of regular support.

## Approach to Dependencies of Airflow

Airflow uses constraints to ensure repeatable installations, while allowing for flexible dependency upgrades.  Upper bounds are used sparingly, with justifications provided.

*   Dependencies maintained in ``pyproject.toml``.
*   Important dependencies that are upper-bound include:
    *   `SQLAlchemy`
    *   `Alembic`
    *   `Flask`
    *   `werkzeug`
    *   `celery`
    *   `kubernetes`
*   Dependencies for Airflow Providers and extras are maintained in `provider.yaml`. By default, not upper-bound.

<!-- START Contributing, please keep comment here to allow auto update of PyPI readme.md -->

## Contributing

Contribute to Apache Airflow by following the guidelines in the [contributors' guide](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst).  Get started quickly with the [contribution quickstart](https://github.com/apache/airflow/blob/main/contributing-docs/03_contributors_quick_start.rst). Official Docker (container) images for Apache Airflow are described in [images](https://github.com/apache/airflow/blob/main/dev/breeze/doc/ci/02_images.md).

<!-- END Contributing, please keep comment here to allow auto update of PyPI readme.md -->
<!-- START Who uses Apache Airflow, please keep comment here to allow auto update of PyPI readme.md -->

## Voting Policy

*   Commits require a +1 vote from a committer who is not the author.
*   AIP voting considers both PMC and committer votes.

## Who Uses Apache Airflow?

Airflow is used by approximately 500 organizations, with more using it "in the wild" ([INTHEWILD.md](https://github.com/apache/airflow/blob/main/INTHEWILD.md)).
If you use Airflow, feel free to make a PR to add your organization to the list.

<!-- END Who uses Apache Airflow, please keep comment here to allow auto update of PyPI readme.md -->
<!-- START Who maintains Apache Airflow, please keep comment here to allow auto update of PyPI readme.md -->

## Who Maintains Apache Airflow?

Airflow is a community-driven project. The [core committers/maintainers](https://people.apache.org/committers-by-project.html#airflow) are responsible for reviewing PRs and steering new feature requests.
Review the Apache Airflow [committer requirements](https://github.com/apache/airflow/blob/main/COMMITTERS.rst#guidelines-to-become-an-airflow-committer) if you'd like to become a maintainer.

<!-- END Who maintains Apache Airflow, please keep comment here to allow auto update of PyPI readme.md -->

## What Goes into the Next Release?

The decision on when PRs are merged into specific releases depend on various factors. More details are explained in detail in this README under the [Semantic versioning](#semantic-versioning) chapter.
More context and **FAQ** about the patchlevel release can be found in the
[What goes into the next release](dev/WHAT_GOES_INTO_THE_NEXT_RELEASE.md) document in the `dev` folder of the
repository.

## Can I use the Apache Airflow Logo in My Presentation?

Yes, as long as you adhere to the Apache Foundation [trademark policies](https://www.apache.org/foundation/marks/#books) and the Apache Airflow [Brandbook](https://cwiki.apache.org/confluence/display/AIRFLOW/Brandbook). The most up-to-date logos are found in [this repo](https://github.com/apache/airflow/tree/main/airflow-core/docs/img/logos/) and on the Apache Software Foundation [website](https://www.apache.org/logos/about.html).

## Links

*   [Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
*   [Chat](https://s.apache.org/airflow-slack)
*   [Community Information](https://airflow.apache.org/community/)

## Sponsors

<a href="https://astronomer.io"><img src="https://assets2.astronomer.io/logos/logoForLIGHTbackground.png" alt="astronomer.io" width="250px"></a>
<a href="https://aws.amazon.com/opensource/"><img src="https://github.com/apache/airflow/blob/main/providers/amazon/docs/integration-logos/AWS-Cloud-alt_light-bg@4x.png?raw=true" alt="AWS OpenSource" width="130px"></a>