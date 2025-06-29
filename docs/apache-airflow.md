# Apache Airflow: Orchestrate Your Workflows with Code

**Apache Airflow is a powerful platform that allows you to programmatically author, schedule, and monitor workflows, making data pipelines more maintainable and reliable.**  Get started with Apache Airflow ([original repo](https://github.com/apache/airflow)) today and streamline your data orchestration.

## Key Features

*   **Dynamic:** Define pipelines in code for dynamic DAG generation and parameterization.
*   **Extensible:** Utilize a wide range of built-in operators and easily extend Airflow to fit your specific needs.
*   **Flexible:** Leverage the power of the [Jinja](https://jinja.palletsprojects.com) templating engine for rich customizations.
*   **Scalable:** Orchestrate complex workflows with a robust scheduler and worker architecture.
*   **Monitoring:** Visualize pipeline runs, monitor progress, and troubleshoot issues through a user-friendly UI.

## Installation & Getting Started

For a comprehensive guide to setting up your local development environment and installing Apache Airflow, please refer to the [INSTALLING.md](INSTALLING.md) file.

You can also visit the official Airflow website documentation (latest **stable** release) for help with
[installing Airflow](https://airflow.apache.org/docs/apache-airflow/stable/installation/),
[getting started](https://airflow.apache.org/docs/apache-airflow/stable/start.html), or walking
through a more complete [tutorial](https://airflow.apache.org/docs/apache-airflow/stable/tutorial/).

> Note: If you're looking for documentation for the main branch (latest development branch): you can find it on [s.apache.org/airflow-docs](https://s.apache.org/airflow-docs/).

## Installing from PyPI

Install Apache Airflow from PyPI using `pip`.

```bash
pip install 'apache-airflow==3.0.2' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.2/constraints-3.9.txt"
```

## Official source code
For more information on Airflow Improvement Proposals (AIPs), visit
the [Airflow Wiki](https://cwiki.apache.org/confluence/display/AIRFLOW/Airflow+Improvement+Proposals).

Documentation for dependent projects like provider distributions, Docker image, Helm Chart, you'll find it in [the documentation index](https://airflow.apache.org/docs/).

Official source code releases are cryptographically signed and available for download from the [ASF Distribution Directory](https://downloads.apache.org/airflow).

## Project Focus

Airflow is an excellent choice for workflows that are relatively static and change gradually. Other notable workflow management tools include [Luigi](https://github.com/spotify/luigi), [Oozie](https://oozie.apache.org/) and [Azkaban](https://azkaban.github.io/).

Airflow is commonly used for data processing, with a focus on idempotent tasks that avoid creating duplicate data. For high-volume, data-intensive operations, it's recommended to delegate to specialized external services.

While Airflow isn't a streaming solution, it's often used to process real-time data in batches.

## User Interface
Airflow provides a rich user interface with a number of features:

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

As of Airflow 2.0.0, Airflow follows a strict [SemVer](https://semver.org/) approach.

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

## Support for Python and Kubernetes versions

Airflow's Python and Kubernetes version support is governed by the official release schedules of Python and Kubernetes.

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

Airflow's dependency management strategy balances the need for application stability with the flexibility of using the latest versions of dependencies.

### Approach for dependencies for Airflow Core

Certain dependencies are upper-bound to ensure stability, including:

*   `SQLAlchemy`
*   `Alembic`
*   `Flask`
*   `werkzeug`
*   `celery`
*   `kubernetes`

### Approach for dependencies in Airflow Providers and extras
In general, providers and extras do not upper-bound dependencies.

## Contributing

Contribute to the project by following the [contributors' guide](https://github.com/apache/airflow/blob/main/contributing-docs/README.rst), or for a quick start check out the [contribution quickstart](https://github.com/apache/airflow/blob/main/contributing-docs/03_contributors_quick_start.rst)

## Voting Policy

Airflow commits require a +1 vote from a committer, and AIP voting includes both PMC and committer votes.

## Who uses Apache Airflow?

Airflow is used by approximately 500 organizations, listed [in the wild](https://github.com/apache/airflow/blob/main/INTHEWILD.md).

## Who maintains Apache Airflow?

Airflow is community-driven, with core committers and maintainers responsible for code reviews and feature direction.

## What goes into the next release?

The release process is governed by the [Semver](https://semver.org/) versioning scheme and explained in
[Airflow release process](https://airflow.apache.org/docs/apache-airflow/stable/release-process.html)
in detail in the README under the [Semantic versioning](#semantic-versioning) chapter.

## Can I use the Apache Airflow logo in my presentation?

Yes! Use the official logos from [this repo](https://github.com/apache/airflow/tree/main/airflow-core/docs/img/logos/) or the Apache Software Foundation [website](https://www.apache.org/logos/about.html), and follow the Apache Foundation [trademark policies](https://www.apache.org/foundation/marks/#books) and the Apache Airflow [Brandbook](https://cwiki.apache.org/confluence/display/AIRFLOW/Brandbook).

## Links

*   [Documentation](https://airflow.apache.org/docs/apache-airflow/stable/)
*   [Chat](https://s.apache.org/airflow-slack)
*   [Community Information](https://airflow.apache.org/community/)

## Sponsors

The CI infrastructure for Apache Airflow has been sponsored by:

<!-- Ordered by most recently "funded" -->

<a href="https://astronomer.io"><img src="https://assets2.astronomer.io/logos/logoForLIGHTbackground.png" alt="astronomer.io" width="250px"></a>
<a href="https://aws.amazon.com/opensource/"><img src="https://github.com/apache/airflow/blob/main/providers/amazon/docs/integration-logos/AWS-Cloud-alt_light-bg@4x.png?raw=true" alt="AWS OpenSource" width="130px"></a>