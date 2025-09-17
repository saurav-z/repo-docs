# Apache Airflow: Automate, Schedule, and Monitor Workflows

**Apache Airflow** is a powerful platform for programmatically authoring, scheduling, and monitoring data pipelines, making complex workflows manageable and efficient.  [Learn more at the original repo](https://github.com/apache/airflow)!

## Key Features:

*   **Dynamic Workflows:** Define pipelines as code for versioning, testing, and collaboration.
*   **Extensible Architecture:** Leverage built-in operators and extend Airflow to fit your specific needs.
*   **Flexible Templating:** Utilize the Jinja templating engine for rich customizations.
*   **Scalable Scheduling:** Execute tasks efficiently across a distributed worker pool.
*   **Rich User Interface:** Visualize pipelines, monitor progress, and troubleshoot with ease.

## Project Focus

Airflow excels with workflows that are mostly static and slowly changing, clarifying the unit of work and continuity.  It is often used for data processing, but tasks should ideally be idempotent.

## Principles

*   **Dynamic:** Pipelines are defined in code, enabling dynamic dag generation and parameterization.
*   **Extensible:** The Airflow framework includes a wide range of built-in operators and can be extended to fit your needs.
*   **Flexible:** Airflow leverages the [**Jinja**](https://jinja.palletsprojects.com) templating engine, allowing rich customizations.

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

Explore the official Airflow documentation for installation, getting started guides, and tutorials:  [Official Airflow Documentation](https://airflow.apache.org/docs/apache-airflow/stable/installation/), [getting started](https://airflow.apache.org/docs/apache-airflow/stable/start.html), or walking
through a more complete [tutorial](https://airflow.apache.org/docs/apache-airflow/stable/tutorial/).

> Note: If you're looking for documentation for the main branch (latest development branch): you can find it on [s.apache.org/airflow-docs](https://s.apache.org/airflow-docs/).

For more information on Airflow Improvement Proposals (AIPs), visit
the [Airflow Wiki](https://cwiki.apache.org/confluence/display/AIRFLOW/Airflow+Improvement+Proposals).

Documentation for dependent projects like provider distributions, Docker image, Helm Chart, you'll find it in [the documentation index](https://airflow.apache.org/docs/).

## Installing from PyPI

Install Airflow from PyPI with the `apache-airflow` package. Due to the application-like nature of Airflow, proper installation requires constraints files to ensure repeatable results.

1.  **Installing Airflow:**

    ```bash
    pip install 'apache-airflow==3.0.6' \
     --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.6/constraints-3.10.txt"
    ```

2.  **Installing with extras (e.g., postgres, google):**

    ```bash
    pip install 'apache-airflow[postgres,google]==3.0.6' \
     --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.6/constraints-3.10.txt"
    ```

    For provider distributions, check [providers](http://airflow.apache.org/docs/apache-airflow-providers/index.html).

## Installation

For detailed setup and installation instructions, refer to the [INSTALLING.md](INSTALLING.md) file.

## Official Source Code

Apache Airflow is an Apache Software Foundation (ASF) project, with official releases adhering to the ASF Release Policy and available for download from the ASF Distribution Directory.

## Convenience Packages

Install and use Airflow via these convenience methods:

*   **PyPI:**  Install using `pip`.
*   **Docker Images:** Deploy Airflow using Docker for Kubernetes, Helm Charts, etc. [Docker Images](https://hub.docker.com/r/apache/airflow).  Learn about using, customizing, and extending the images in the [Latest docs](https://airflow.apache.org/docs/docker-stack/index.html).
*   **GitHub Tags:** Access the source code used to generate official packages via Git tags. [Tags in GitHub](https://github.com/apache/airflow/tags).

## User Interface

Airflow's UI provides comprehensive workflow monitoring and management:

*   **DAGs:** Overview of all DAGs.
*   **Assets:** Overview of Assets with dependencies.
*   **Grid:** DAG representation across time.
*   **Graph:** DAG dependencies and status visualization.
*   **Home:** Summary statistics.
*   **Backfill:** DAG backfilling.
*   **Code:** View DAG source code.

  ![DAGs](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/dags.png)

  ![Asset Dependencies](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/assets_graph.png)

  ![Grid](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/grid.png)

  ![Graph](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/graph.png)

  ![Home](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/home.png)

  ![Backfill](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/backfill.png)

  ![Code](https://raw.githubusercontent.com/apache/airflow/main/airflow-core/docs/img/ui-dark/code.png)


## Semantic Versioning

Airflow 2.0.0 and later uses a strict SemVer approach, defining specific versioning rules for various packages.

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

Limited support versions will be supported with security and critical bug fix only.
EOL versions will not get any fixes nor support.
We always recommend that all users run the latest available minor release for whatever major version is in use.
We **highly** recommend upgrading to the latest Airflow major release at the earliest convenient time and before the EOL date.

## Support for Python and Kubernetes versions

Airflow follows specific guidelines for Python and Kubernetes support, based on their respective release schedules.

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

* `SQLAlchemy`: upper-bound to specific MINOR version (SQLAlchemy is known to remove deprecations and
   introduce breaking changes especially that support for different Databases varies and changes at
   various speed)
* `Alembic`: it is important to handle our migrations in predictable and performant way. It is developed
   together with SQLAlchemy. Our experience with Alembic is that it very stable in MINOR version
* `Flask`: We are using Flask as the back-bone of our web UI and API. We know major version of Flask
   are very likely to introduce breaking changes across those so limiting it to MAJOR version makes sense
* `werkzeug`: the library is known to cause problems in new versions. It is tightly coupled with Flask
   libraries, and we should update them together
* `celery`: Celery is a crucial component of Airflow as it used for CeleryExecutor (and similar). Celery
   [follows SemVer](https://docs.celeryq.dev/en/stable/contributing.html?highlight=semver#versions), so
   we should upper-bound it to the next MAJOR version. Also, when we bump the upper version of the library,
   we should make sure Celery Provider minimum Airflow version is updated.
* `kubernetes`: Kubernetes is a crucial component of Airflow as it is used for the KubernetesExecutor
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

Contribute to Apache Airflow by following the [contributors' guide](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst), and also check out the [contribution quickstart](https://github.com/apache/airflow/blob/main/contributing-docs/03_contributors_quick_start.rst)

Official Docker (container) images for Apache Airflow are described in [images](https://github.com/apache/airflow/blob/main/dev/breeze/doc/ci/02_images.md).

## Voting Policy

*   Commits need a +1 vote from a committer who is not the author
*   When we do AIP voting, both PMC member's and committer's `+1s` are considered a binding vote.

## Who uses Apache Airflow?

Many organizations use Apache Airflow, find out about around 500 of them [in the wild](https://github.com/apache/airflow/blob/main/INTHEWILD.md).

If you use Airflow - feel free to make a PR to add your organisation to the list.

## Who maintains Apache Airflow?

Airflow is a community-driven project, with [core committers/maintainers](https://people.apache.org/committers-by-project.html#airflow) responsible for reviewing and merging pull requests.  Learn about becoming a maintainer by reviewing the Apache Airflow [committer requirements](https://github.com/apache/airflow/blob/main/COMMITTERS.rst#guidelines-to-become-an-airflow-committer).

## What goes into the next release?

For details on what goes into a release check the [dev/WHAT_GOES_INTO_THE_NEXT_RELEASE.md](dev/WHAT_GOES_INTO_THE_NEXT_RELEASE.md) document in the `dev` folder of the repository.

## Can I use the Apache Airflow logo in my presentation?

Yes, provided you follow the Apache Foundation [trademark policies](https://www.apache.org/foundation/marks/#books) and the Apache Airflow [Brandbook](https://cwiki.apache.org/confluence/display/AIRFLOW/Brandbook). Use the latest logos found in [this repo](https://github.com/apache/airflow/tree/main/airflow-core/docs/img/logos/) and on the Apache Software Foundation [website](https://www.apache.org/logos/about.html).

## Links

*   [Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
*   [Chat](https://s.apache.org/airflow-slack)
*   [Community Information](https://airflow.apache.org/community/)

## Sponsors

This project is sponsored by:

<!-- Ordered by most recently "funded" -->

<a href="https://astronomer.io"><img src="https://assets2.astronomer.io/logos/logoForLIGHTbackground.png" alt="astronomer.io" width="250px"></a>
<a href="https://aws.amazon.com/opensource/"><img src="https://github.com/apache/airflow/blob/main/providers/amazon/docs/integration-logos/AWS-Cloud-alt_light-bg@4x.png?raw=true" alt="AWS OpenSource" width="130px"></a>